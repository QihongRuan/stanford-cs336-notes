## Scaling Laws & Compute-Optimal Training
### Q: The notes highlight that Chinchilla's main correction to earlier work (like Kaplan et al.) was properly accounting for the cosine learning rate schedule. Why, precisely, does using intermediate checkpoints from a single long training run with a cosine schedule lead to incorrect scaling law estimates?

This is a subtle but critical point that gets to the heart of why fitting scaling laws is so computationally demanding. The error stems from a mismatch between the training dynamics of a truncated run and a run that was *designed* to be short from the outset.

**The Goal & The Flawed Shortcut**

The goal of this analysis is to find the loss `L(N, D)` as a function of model size `N` and dataset size `D`. To do this, we need to measure the final loss for many different pairs of `(N, D)`.

The flawed shortcut, used in some early analyses, is to train one large model for a very long time (many tokens) and save checkpoints along the way. The assumption is that a checkpoint at `D_i` tokens represents the final state of a model trained on a dataset of size `D_i`. This would be a huge computational saving, as one long run could provide data for dozens of `(N, D)` points.

**Why the Cosine Schedule Breaks This Shortcut**

Modern LLMs are almost universally trained with a cosine annealing learning rate schedule. The learning rate `η` at step `t` is given by:

$$
\eta(t) = \eta_{min} + \frac{1}{2} (\eta_{max} - \eta_{min}) \left(1 + \cos\left(\frac{\pi t}{T_{total}}\right)\right)
$$

The crucial variable here is `T_{total}`, the total number of training steps planned for the entire run. The entire *shape* of the learning rate curve depends on this final horizon.

Let's compare two scenarios:

1.  **Full Run:** We train a model for `T_{total} = 100,000` steps. The LR schedule is a slow, gradual decay over all 100k steps.
2.  **Truncated Run:** We take the model from Scenario 1 at step `t = 20,000`. The flawed assumption is that this model is representative of a model trained optimally for just 20,000 steps.
3.  **Correct Short Run:** We train a new model from scratch with `T_{total} = 20,000`. Its cosine schedule will be much more aggressive, decaying from `η_max` to `η_min` over just 20k steps.

**The Mismatch in Dynamics**

The model from the truncated run (Scenario 2) is "undertrained" relative to its learning rate. At step 20,000, its learning rate is still very high because it's "expecting" another 80,000 steps of training. In contrast, the model from the correct short run (Scenario 3) has already gone through its entire high-to-low LR cycle. It has spent more of its (short) life at lower learning rates, allowing it to settle into a better minimum for that specific training duration.

Therefore, `Loss(Scenario 2)` will almost certainly be higher than `Loss(Scenario 3)`. Using the loss from the truncated run as a data point for `L(N, D=20k_steps)` would be an overestimate of the model's true capability for that data budget, biasing the resulting scaling law.

**Chinchilla's Solution (and its cost)**

The Chinchilla authors recognized this flaw. Their solution was simple but expensive: for each data point `(N_i, D_i)` they wanted to measure, they trained a separate model from scratch with a full cosine schedule optimized for that specific `D_i` (i.e., for `T_{total} = D_i / \text{batch_size}`). This ensures that every measured loss value represents a properly converged model for its given budget. This increases the computational cost of the analysis quadratically, but yields a much more accurate result.

**A More Modern, Efficient Alternative: WSD Schedules**

Later work (e.g., MiniCPM, DeepSeek) proposed a compromise using a **Warm-up Stable Decay (WSD)** or trapezoidal learning rate schedule.

```python
def wsd_schedule(step, warmup_steps, total_steps, decay_steps):
    stable_steps = total_steps - warmup_steps - decay_steps
    if step < warmup_steps:
        # Linear warmup
        return (step / warmup_steps) * max_lr
    elif step < warmup_steps + stable_steps:
        # Stable phase
        return max_lr
    else:
        # Cosine decay over the last part
        decay_progress = (step - warmup_steps - stable_steps) / decay_steps
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_progress))
```

The key insight is that during the long "stable" phase, the training dynamics are consistent. One can perform a single long run, and then for each desired data budget `D_i`, rewind to the checkpoint at the end of the stable phase for that budget and apply the *final, short decay phase*. This largely avoids the cosine schedule mismatch problem while being much cheaper than training dozens of models from scratch.

### Q: Chinchilla's "Method 2" uses "IsoFLOP curves" to find the optimal model and data size. Can you provide a step-by-step algorithmic derivation of this method and the intuition behind it?

The IsoFLOPs method is an elegant and robust way to determine the optimal trade-off between model size and data size for a fixed compute budget. It directly answers the question: "If I have a budget of `C` FLOPs, what is the best model size `N` and number of training tokens `D` to use?"

**The Core Constraint: The Compute Equation**

First, we need a model for the total training compute `C` (in FLOPs). A widely used approximation for a Transformer is:

$$
C \approx 6 \times N \times D
$$

Where:

- `N` is the number of non-embedding parameters.
- `D` is the number of training tokens.
- The factor of 6 comes from: `2` for the matrix multiply in the forward pass (`2 * N`), `4` for the corresponding backward pass (`4 * N`), totaling `6N` FLOPs per token.

An "IsoFLOP curve" (iso meaning "same") is the set of all pairs `(N, D)` that result in the same total compute `C`. On a plot of `D` vs. `N`, this is a hyperbola: `D = C / (6N)`.

**The IsoFLOP Algorithm**

The algorithm proceeds in two main phases: (1) Finding the optimal point for several fixed budgets, and (2) Fitting a scaling law to those optimal points.

**Phase 1: Finding Optimal `(N, D)` for Fixed Budgets `C`**

1.  **Select Compute Budgets:** Choose a set of exponentially spaced compute budgets to explore, e.g., `C_1 = 10^20`, `C_2 = 10^21`, `C_3 = 10^22` FLOPs.

2.  **For each budget `C_k`:**
    a. **Generate IsoFLOP Points:** Choose several points `(N_i, D_i)` that lie on the IsoFLOP curve for `C_k`. A good strategy is to vary `N` exponentially.
       For example, for `C_k = 10^{21}`:
       - Point 1 (Small Model, Lots of Data): `N_1 = 70M`, so `D_1 = 10^{21} / (6 * 70M) ≈ 2.4T` tokens.
       - Point 2 (Medium Model, Medium Data): `N_2 = 350M`, so `D_2 = 10^{21} / (6 * 350M) ≈ 480B` tokens.
       - Point 3 (Large Model, Less Data): `N_3 = 1.75B`, so `D_3 = 10^{21} / (6 * 1.75B) ≈ 95B` tokens.

    b. **Train Models:** Train a separate model from scratch for each point `(N_i, D_i)`. Each model of size `N_i` is trained for `D_i` tokens. Record the final validation loss `L_i`.

    c. **Find the Minimum:** Plot the resulting losses `L_i` against the model sizes `N_i`. This will produce a characteristic U-shaped curve. The bottom of the "U" represents the best trade-off. Find the model size `N_{opt}(C_k)` that minimizes the loss for this compute budget `C_k`. The corresponding optimal number of tokens is `D_{opt}(C_k) = C_k / (6 \times N_{opt}(C_k))`.

**Phase 2: Fitting the Scaling Law**

3.  **Collect Optimal Points:** After repeating Phase 1 for all chosen budgets, you will have a set of optimal allocations:
    - `(C_1, N_{opt}(C_1), D_{opt}(C_1))`
    - `(C_2, N_{opt}(C_2), D_{opt}(C_2))`
    - `...`

4.  **Fit Power Laws:** The hypothesis is that the optimal model size and data size scale as a power law of the compute budget:
    $$
    N_{opt}(C) = K_N \cdot C^a
    $$
    $$
    D_{opt}(C) = K_D \cdot C^b
    $$
    To find the exponents `a` and `b`, we perform a linear regression in log-log space:
    $$
    \log(N_{opt}) = a \cdot \log(C) + \log(K_N)
    $$
    $$
    \log(D_{opt}) = b \cdot \log(C) + \log(K_D)
    $$
    The slopes of these lines give us the desired scaling exponents `a` and `b`.

**Pseudocode for the entire process:**

```python
budgets = [1e20, 1e21, 1e22, 1e23] # FLOPs
optimal_points = []

for C in budgets:
    # Phase 1: Find optimum for this budget C
    model_sizes_to_test = [70e6, 350e6, 1.75e9, 8e9] # Example N values
    results_for_C = []
    for N in model_sizes_to_test:
        D = C / (6 * N)
        # Train model of size N on D tokens to convergence
        final_loss = train_model(N, D)
        results_for_C.append({'N': N, 'loss': final_loss})

    # Find the N that resulted in the minimum loss for this C
    # (In practice, you'd fit a quadratic or other curve to the points)
    best_result = min(results_for_C, key=lambda x: x['loss'])
    N_opt = best_result['N']
    D_opt = C / (6 * N_opt)
    optimal_points.append({'C': C, 'N_opt': N_opt, 'D_opt': D_opt})

# Phase 2: Fit scaling law to the optimal points
log_C = [math.log(p['C']) for p in optimal_points]
log_N_opt = [math.log(p['N_opt']) for p in optimal_points]
log_D_opt = [math.log(p['D_opt']) for p in optimal_points]

# Perform linear regression: log_N_opt = a * log_C + const
a = linear_regression(log_C, log_N_opt).slope
# Perform linear regression: log_D_opt = b * log_C + const
b = linear_regression(log_C, log_D_opt).slope

print(f"Optimal scaling exponents: a={a:.2f}, b={b:.2f}")
```

**Intuition and Result**

The U-shape of the loss curve for a fixed budget makes intuitive sense:

- **Too small `N` (left side of U):** The model lacks the capacity to learn from the vast amount of data (`D` is huge). It becomes the bottleneck.
- **Too large `N` (right side of U):** The model is too large for the small amount of data it's shown (`D` is small). It is severely under-trained and cannot converge to a good solution.

The IsoFLOPs method systematically finds the "sweet spot" at the bottom of this U for various scales. Chinchilla's key finding using this method was that `a ≈ 0.5` and `b ≈ 0.5`. This implies that as you increase your compute budget `C`, you should scale both your model size and your dataset size proportionally to `sqrt(C)`. This was a major departure from previous work which suggested scaling `N` much more aggressively than `D`.

### Q: The notes state that Chinchilla-optimal training may not be what you want for deployment, leading to a trend of "over-training". Can you formalize the trade-off between training and inference costs to show why a smaller, "over-trained" model can be economically superior?

This is a crucial insight that bridges scaling law research with real-world production economics. The "compute-optimal" point found by Chinchilla minimizes the FLOPs required during **training** to reach a certain loss. However, for any successful model, the total compute spent on **inference** over its lifetime can dwarf the one-time training cost.

**Formalizing the Costs**

Let's define two cost components:

1.  **Training Cost (`Cost_train`):** This is the one-time compute budget, which we model as `C = 6ND`.
2.  **Inference Cost (`Cost_inference`):** This is the ongoing cost. For each forward pass, the cost is proportional to the model size `N`. If we expect a model to serve a total of `K` tokens over its lifetime, the total inference cost is:
    $$
    \text{Cost}_{inference} \approx \gamma \cdot K \cdot N
    $$
    where `γ` is a constant representing FLOPs per parameter per token for inference (roughly 2). The key is that inference cost is directly proportional to model size `N`.

The **Total Cost of Ownership (TCO)** of the model is:
$$
\text{TCO}(N, D) = \text{Cost}_{train} + \text{Cost}_{inference} = 6ND + \gamma K N
$$

**The Chinchilla-Optimal Strategy**

Chinchilla's analysis provides a recipe to achieve a target loss `L_target` with the minimum training compute `C_train`. Let's say this recipe calls for a model of size `N_{opt}` trained on `D_{opt}` tokens.

- `C_{opt} = 6 N_{opt} D_{opt}`
- The TCO for this strategy is: `TCO_{opt} = C_{opt} + \gamma K N_{opt}`

**The "Over-training" Strategy**

Now, consider an alternative. What if we choose a **smaller model**, `N' < N_{opt}`?

- **Benefit:** The inference cost will be lower: `Cost'_{inference} = \gamma K N' < \gamma K N_{opt}`.
- **Problem:** To reach the *same target loss* `L_target` with a smaller model, we must compensate for its lower capacity by training it on much more data. Let's call this `D'`.

We can use the Chinchilla scaling law for loss to find the required `D'`:
$$
L(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + E_{ir}
$$
We set `L(N_{opt}, D_{opt}) = L(N', D')` and solve for `D'`:
$$
A \cdot N_{opt}^{-\alpha} + B \cdot D_{opt}^{-\beta} = A \cdot (N')^{-\alpha} + B \cdot (D')^{-\beta}
$$
$$
(D')^{-\beta} = \frac{A}{B} (N_{opt}^{-\alpha} - (N')^{-\alpha}) + D_{opt}^{-\beta}
$$
Since `N' < N_{opt}`, the term `(N_{opt}^{-\alpha} - (N')^{-\alpha})` is negative, but this seems wrong. Let's re-examine the equation. `N' < N_{opt}` means `(N')^{-α} > N_{opt}^{-α}`. So the term is positive. This means `(D')^{-β}` will be larger, which implies `D'` will be smaller. This is the opposite of what we expect.

Let's re-think the logic. The IsoFLOP curves show that for a *fixed* compute budget, moving away from `N_opt` increases loss. To get back to the same loss level with a smaller model, we must *increase* the total compute budget by training for longer.

Let's use a simpler framing. The loss `L` is on a contour line in the `(N, D)` plane. The Chinchilla-optimal point `(N_{opt}, D_{opt})` is the point on this contour that minimizes `C = 6ND`. Any other point `(N', D')` on the same loss contour will have `C' = 6N'D' > C_{opt}`.

So, the "over-training" strategy is:

1.  Pick a smaller model size `N' < N_{opt}`.
2.  Find the data `D'` required to hit the same loss `L_target`. This point `(N', D')` will lie on the same loss iso-line as `(N_{opt}, D_{opt})`.
3.  This new training plan will have a higher training cost: `C' = 6N'D' > C_{opt}`.
4.  The TCO for this new strategy is: `TCO' = C' + \gamma K N'`.

**When is Over-training Worth It?**

We prefer the over-training strategy if `TCO' < TCO_{opt}`:
$$
C' + \gamma K N' < C_{opt} + \gamma K N_{opt}
$$
$$
(C' - C_{opt}) < \gamma K (N_{opt} - N')
$$

Let's analyze this inequality:

- **Left side (`C' - C_{opt}`):** This is the **increase in training cost**. It's a positive, one-time cost.
- **Right side (`γK(N_{opt} - N')`):** This is the **total savings in inference cost** over the model's lifetime. It's also positive.

The inequality shows that over-training is economically superior if the lifetime inference savings outweigh the upfront increase in training cost.

**Numerical Example**

- Suppose Chinchilla-optimal for loss `L` is `N_{opt}=70B`, `D_{opt}=1.4T` tokens. `C_{opt} \propto 70 \times 1.4 = 98`.
- Suppose we can achieve the same loss `L` with a smaller model `N'=50B` by "over-training" on `D'=2.5T` tokens.
- The new training cost is `C' \propto 50 \times 2.5 = 125`. The training cost increased by `125 - 98 = 27` units.
- The inference cost is proportional to `N`. The savings are proportional to `N_{opt} - N' = 70 - 50 = 20`.
- The trade-off becomes: `27 < (\text{Inference Cost Factor}) \times 20`.

If the model is a research prototype used a few times, the inference factor is small, and the Chinchilla-optimal model is cheaper overall. But if it's a production model like Llama 3 serving billions of queries (`K` is huge), the inference factor is massive, the inequality holds, and it is far more economical to pay the extra training cost to get a smaller, cheaper-to-run final model. This is why we see models like Llama 3 being trained on over 15T tokens, far exceeding the original Chinchilla ratio of 20 tokens/parameter.