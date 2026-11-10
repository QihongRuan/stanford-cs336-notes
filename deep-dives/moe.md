## Mixture of Experts: Routing & Load Balancing

### Q: The notes say Top-K routing is "non-differentiable," so how are the router's parameters trained at all? What gradient signal does the router receive?

This is a crucial point. The `TopK` operation, which selects the indices of the $K$ largest values from a vector of scores, is a discrete selection. Its gradient is zero almost everywhere, and undefined where ties occur. A naive implementation would block gradients from flowing back to the router parameters for any expert that wasn't selected, providing no signal on how to improve.

The solution is to use a **Straight-Through Estimator (STE)**. In the forward pass, we perform the hard, non-differentiable `TopK` selection. In the backward pass, we "pretend" this selection was a simple multiplication by a 0/1 mask and allow gradients to pass through the selected "gates" as if they were always open.

Let's break down the math.

**1. Forward Pass:**

For a single token with hidden state $x \in \mathbb{R}^H$, the router computes scores for $N$ experts. The router has a weight matrix $W_g \in \mathbb{R}^{H \times N}$.

1.  **Logits:** Compute raw scores (logits) for each expert.
    $$ z = x W_g \quad (z \in \mathbb{R}^N) $$
2.  **Probabilities:** Apply softmax to get normalized probabilities.
    $$ p = \text{Softmax}(z) \quad (p \in \mathbb{R}^N, \sum p_i = 1) $$
3.  **Gating Mask:** Identify the top $K$ probabilities and create a sparse binary mask.
    $$ M = \text{TopK_Mask}(p, K) \quad (M \in \{0, 1\}^N, \sum M_i = K) $$
4.  **Gating Weights:** The final weight for each expert is its probability, but only if it was selected. This is the STE part: we use the "soft" probability value $p_i$ for the "hard" choice $M_i=1$.
    $$ g_i = M_i \cdot p_i $$
5.  **Final Output:** The token's output is a weighted sum of the outputs from the selected experts.
    $$ y = \sum_{i=1}^N g_i \cdot \text{Expert}_i(x) = \sum_{i \in \text{TopK}} p_i \cdot \text{Expert}_i(x) $$

```python
# Simplified forward pass for one token
def forward_moe_layer(x, experts, router_weights, K):
    # x: [H], router_weights: [H, N]
    logits = x @ router_weights  # [N]
    probs = softmax(logits)      # [N]

    # Non-differentiable step
    top_k_probs, top_k_indices = torch.topk(probs, K)

    # Create a mask for the backward pass (for clarity)
    mask = torch.zeros_like(probs)
    mask.scatter_(0, top_k_indices, 1) # mask is [0, 1, 0, 0, 1, ...] for K=2

    # Gating weights (STE: use soft probs for hard choices)
    gating_weights = probs * mask # This is where the "lie" happens for backprop
    # In a real implementation, you'd just use top_k_probs and top_k_indices
    # to avoid computing on all experts.

    # Compute expert outputs
    final_output = 0
    for i in top_k_indices:
        # The gradient will flow back through here
        expert_output = experts[i](x)
        # And through here, scaled by the probability
        final_output += probs[i] * expert_output

    return final_output
```

**2. Backward Pass (The "Lie"):**

Let $L$ be the final model loss. We need to compute $\frac{\partial L}{\partial W_g}$. Using the chain rule, the gradient flows "through" the selected experts. For a single expert $i$ that was in the Top-K set (i.e., $M_i=1$):

$$ \frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z_i} = \frac{\partial L}{\partial y} \frac{\partial}{\partial z_i} \left( p_i \cdot \text{Expert}_i(x) \right) $$

Assuming the expert's computation $\text{Expert}_i(x)$ does not depend on the router's logits $z$, this simplifies to:

$$ \frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial y} \cdot \text{Expert}_i(x) \cdot \frac{\partial p_i}{\partial z_i} $$

The crucial part is that for any expert $j$ *not* in the Top-K set (i.e., $M_j=0$), the gating weight $g_j$ was 0, so its contribution to the output $y$ was 0. Therefore, $\frac{\partial L}{\partial z_j} = 0$.

**The Pitfall:** The router receives **zero gradient signal** for the experts it did not select. It gets feedback on how to adjust probabilities *among the winners*, but it never learns that it should have picked a different expert entirely. This can lead to a "rich get richer" dynamic where a few experts are always chosen, while others "die" from lack of training signals. This is precisely why auxiliary load balancing losses are not just helpful, but essential for stable MoE training.

---

### Q: How does the Switch Transformer's load balancing loss, $L_{aux} = \alpha \sum_{i=1}^{N} f_i P_i$, mathematically encourage experts to be used more evenly?

This auxiliary loss is a clever heuristic that pushes the router away from imbalanced expert assignments. To understand how, we must derive its gradient with respect to the router's logits.

First, let's define the terms for a batch of $T$ tokens and $N$ experts:

- **$p_{i,t}$**: The softmax probability from the router for expert $i$ and token $t$.
- **$P_i = \frac{1}{T} \sum_{t=1}^{T} p_{i,t}$**: The average router probability assigned to expert $i$ over the batch. This is the router's *intention* to use expert $i$.
- **$c_{i,t} \in \{0, 1\}$**: An indicator that is 1 if expert $i$ was chosen for token $t$ (i.e., was in the Top-K), and 0 otherwise.
- **$f_i = \frac{1}{T} \sum_{t=1}^{T} c_{i,t}$**: The fraction of tokens in the batch that were *actually dispatched* to expert $i$. This is the expert's *utilization*.

The auxiliary loss is:
$$ L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i $$
*(Note: The original paper includes a factor of N for scaling, so we'll include it here.)*

The key trick is that during backpropagation for this specific loss term, the expert utilization $f_i$ (which comes from the non-differentiable `TopK` choice) is treated as a **constant**. The gradient is only computed with respect to the differentiable probabilities $P_i$.

Let's find the gradient of $L_{aux}$ with respect to a single router logit $z_{k,t}$ (for expert $k$, token $t$). We use the chain rule:

1.  **Gradient w.r.t. average probability $P_j$**:
    $$ \frac{\partial L_{aux}}{\partial P_j} = \alpha N f_j $$

2.  **Gradient of $P_j$ w.r.t. single probability $p_{k,t}$**:
    $$ \frac{\partial P_j}{\partial p_{k,t}} = \frac{\partial}{\partial p_{k,t}} \left( \frac{1}{T} \sum_{t'=1}^{T} p_{j,t'} \right) = \frac{1}{T} \delta_{jk} $$
    where $\delta_{jk}$ is the Kronecker delta (1 if $j=k$, 0 otherwise).

3.  **Gradient w.r.t. single probability $p_{k,t}$**:
    $$ \frac{\partial L_{aux}}{\partial p_{k,t}} = \sum_{j=1}^{N} \frac{\partial L_{aux}}{\partial P_j} \frac{\partial P_j}{\partial p_{k,t}} = \frac{\partial L_{aux}}{\partial P_k} \frac{\partial P_k}{\partial p_{k,t}} = (\alpha N f_k) \left( \frac{1}{T} \right) = \frac{\alpha N f_k}{T} $$

4.  **Gradient w.r.t. logit $z_{k,t}$**:
    The gradient of a loss with respect to softmax logits $z_k$ can be expressed in terms of the gradient with respect to the softmax outputs $p_k$:
    $$ \frac{\partial L}{\partial z_k} = \sum_{j=1}^{N} \frac{\partial L}{\partial p_j} \frac{\partial p_j}{\partial z_k} = \sum_{j=1}^{N} \frac{\partial L}{\partial p_j} p_j(\delta_{jk} - p_k) = p_k \left( \frac{\partial L}{\partial p_k} - \sum_{j=1}^{N} \frac{\partial L}{\partial p_j} p_j \right) $$
    Substituting our gradient $\frac{\partial L_{aux}}{\partial p_{j,t}} = \frac{\alpha N f_j}{T}$:
    $$ \frac{\partial L_{aux}}{\partial z_{k,t}} = p_{k,t} \left( \frac{\alpha N f_k}{T} - \sum_{j=1}^{N} \frac{\alpha N f_j}{T} p_{j,t} \right) = \frac{\alpha N p_{k,t}}{T} \left( f_k - \sum_{j=1}^{N} f_j p_{j,t} \right) $$

**Intuition:**
The gradient on logit $z_{k,t}$ is proportional to $f_k - \mathbb{E}_{p_t}[f]$.

- $\boldsymbol{f_k}$ is the actual utilization of expert $k$.
- $\boldsymbol{\sum_{j=1}^{N} f_j p_{j,t}}$ is the expected utilization for this specific token, weighted by the router's current probabilities.

If expert $k$ is over-utilized ($f_k$ is high), then $f_k > \mathbb{E}_{p_t}[f]$, making the gradient positive. During gradient *descent*, the optimizer will push the logit $z_{k,t}$ **down**. This makes the router less likely to assign high probability to the over-utilized expert $k$ for this token in the future.

Conversely, if an expert is under-utilized ($f_k$ is low), the gradient term becomes negative, and the optimizer pushes its logit $z_{k,t}$ **up**, encouraging the router to use it more. This simple, elegant mechanism nudges the router towards a state where all $f_i$ are roughly equal, ensuring all experts are trained.

---

### Q: What is the "Router Z-loss" and why does penalizing the log-sum-exp of logits improve training stability?

The Router Z-loss is an auxiliary loss term designed to prevent the router's logits from growing too large, which is a common source of numerical instability, especially when training with low-precision formats like `bfloat16`.

The loss is defined for each token $t$ as the squared log-sum-exp of its router logits $z_t$, scaled by a small coefficient $\beta$:
$$ L_z = \beta \cdot \left( \log \sum_{i=1}^{N} e^{z_{i,t}} \right)^2 $$
This is then averaged across all tokens in a batch.

**Why does this help?**

1.  **Connection to Softmax and Overflow:** The softmax function, $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$, is sensitive to large logit values. If any $z_i$ becomes very large, $e^{z_i}$ can exceed the maximum representable value for a given floating-point type (e.g., `bfloat16` has a max value of ~3.4e38), causing an `inf` (infinity) result. This leads to `NaN` (Not a Number) in the softmax output and corrupts the gradients, causing the training to "blow up".

2.  **Log-Sum-Exp as a Smooth Maximum:** The function $\text{LSE}(z) = \log \sum_i e^{z_i}$ is a well-known smooth approximation of the maximum function, i.e., $\text{LSE}(z) \approx \max_i(z_i)$. Therefore, the Z-loss is effectively penalizing $(\max_i z_{i,t})^2$. By adding this penalty to the total loss, the optimizer is incentivized to keep the magnitude of all logits, and especially the largest one, under control.

**Mathematical Derivation of the Gradient:**

To see how it works, let's derive the gradient of $L_z$ with respect to a single logit $z_k$.
Let $S = \sum_i e^{z_i}$. Then $L_z = \beta (\log S)^2$.

Using the chain rule:
$$ \frac{\partial L_z}{\partial z_k} = \frac{\partial L_z}{\partial S} \frac{\partial S}{\partial z_k} $$
$$ \frac{\partial S}{\partial z_k} = \frac{\partial}{\partial z_k} \left( \sum_i e^{z_i} \right) = e^{z_k} $$
$$ \frac{\partial L_z}{\partial S} = \beta \cdot 2(\log S) \cdot \frac{1}{S} $$
Combining these:
$$ \frac{\partial L_z}{\partial z_k} = \beta \cdot 2(\log S) \cdot \frac{1}{S} \cdot e^{z_k} = 2\beta \left( \log \sum_i e^{z_i} \right) \left( \frac{e^{z_k}}{\sum_i e^{z_i}} \right) $$
Recognizing the final term as the softmax probability $p_k$, we get:
$$ \frac{\partial L_z}{\partial z_k} = 2\beta \cdot (\text{LSE}(z)) \cdot p_k $$

**Intuition:**
The gradient for logit $z_k$ is always positive (since LSE and $p_k$ are positive). During gradient descent, the optimizer will always push the logits **down**. The magnitude of this "push" is proportional to two things:

1.  **$\text{LSE}(z)$**: The current magnitude of the (smooth) max logit. If the logits are already large, the push is stronger.
2.  **$p_k$**: The router's confidence in expert $k$. The logit for the expert the router is already confident in gets pushed down the most.

This acts as a gentle regularizer, preventing any logit from growing without bound and keeping the router's internal values within a safe numerical range, thus ensuring stable training. It's a simple but highly effective trick for taming MoE models.

---

### Q: How do "fine-grained experts" increase the parameter count for the same computational cost (FLOPs)? A quantitative analysis seems in order.

This is a key insight behind modern MoE architectures like DeepSeek. The trick is to trade the *size* of each expert for the *number* of experts, which allows for a dramatic increase in the total parameter count while keeping the number of *activated* parameters (and thus FLOPs) constant.

Let's analyze this by comparing a dense FFN layer to a fine-grained MoE layer with the same FLOPs budget.

**Definitions:**

- $H$: Model hidden dimension.
- $I$: Intermediate dimension of a standard FFN (e.g., $I = 4H$).
- $T$: Number of tokens in a sequence.
- FLOPs are approximated as $2 \times \text{MACs}$ (Multiply-Accumulate operations).

**1. Baseline: Dense FFN Layer**
A standard FFN consists of an up-projection ($W_{up}: H \to I$) and a down-projection ($W_{down}: I \to H$).

- **Parameters:** The total number of weights is $(H \times I) + (I \times H) = 2HI$.
- **FLOPs:** For each token, we perform two matrix-vector multiplications. The total FLOPs are $T \times (2 \cdot H \cdot I + 2 \cdot I \cdot H) \approx 4THI$.

**2. Fine-Grained MoE Layer (at constant FLOPs)**
Now, we replace the dense FFN with an MoE layer. We want to keep the FLOPs the same as the dense baseline.

- Let the MoE have $N_{fg}$ "fine-grained" experts.
- Each expert has a smaller intermediate dimension, $I_{fg}$.
- The router activates the Top-$K$ experts for each token.

- **FLOPs:** The computation for one token involves $K$ experts. The FLOPs per token are $K \times (2HI_{fg} + 2HI_{fg}) = 4KHI_{fg}$. The total FLOPs are $T \times 4KHI_{fg}$.
- **Equating FLOPs:** To match the dense model's computational cost, we set:
  $$ 4THI = 4TKHI_{fg} \implies I = K \cdot I_{fg} $$
  This is the core constraint: the intermediate dimension of the dense model must equal the *sum* of the intermediate dimensions of the activated experts.

- **Parameters:** The total parameter count of the MoE layer is the sum of parameters across all $N_{fg}$ experts (ignoring the small router).
  $$ \text{Params}_{MoE} = N_{fg} \times (2HI_{fg}) $$
  Substituting $I_{fg} = I/K$ from our FLOPs constraint:
  $$ \text{Params}_{MoE} = N_{fg} \times \left( 2H \frac{I}{K} \right) = \frac{N_{fg}}{K} \cdot (2HI) $$
  Recall that the dense model's parameters were $\text{Params}_{Dense} = 2HI$. Therefore:
  $$ \text{Params}_{MoE} = \frac{N_{fg}}{K} \cdot \text{Params}_{Dense} $$

**The Payoff:**
For the same computational cost as a dense layer, an MoE layer has $\frac{N_{fg}}{K}$ times more parameters.

**Concrete Example:**
Let's design an MoE to replace a dense FFN where $H=4096$ and $I=4H=16384$.

- **Dense FFN:**
  - FLOPs per token $\propto I = 16384$.
  - Parameters $\propto 2HI = 2 \cdot 4096 \cdot 16384 \approx 134M$.

- **Fine-Grained MoE:** Let's use $K=2$ active experts (a common choice).
  - To match FLOPs, we need $K \cdot I_{fg} = I \implies 2 \cdot I_{fg} = 16384 \implies I_{fg} = 8192$. Each expert is half the size of the original dense FFN.
  - Now, let's say we use $N_{fg}=8$ total experts (like Mixtral 8x7B).
  - **Total Parameters:** $\text{Params}_{MoE} = \frac{N_{fg}}{K} \cdot \text{Params}_{Dense} = \frac{8}{2} \times 134M = 4 \times 134M = 536M$.

By replacing the dense layer with an 8-expert MoE (activating 2 per token), we have **quadrupled the parameter count** of that layer while keeping the inference FLOPs **identical**. This allows the model to have a much larger capacity for storing knowledge without increasing the computational cost of a forward pass.