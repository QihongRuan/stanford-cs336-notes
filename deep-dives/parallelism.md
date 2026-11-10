## Parallelism Deep Dive: Data, Tensor, Pipeline, and FSDP

### Q: FSDP (ZeRO-3) is often preferred for its memory savings, but it seems to have 1.5x the communication volume of standard Data Parallelism (`all_reduce`). Why is it still considered highly efficient in practice?

This is a crucial observation. While FSDP (Fully Sharded Data Parallel, equivalent to ZeRO-3) does transfer more data, its efficiency comes from **overlapping communication with computation**. The raw communication volume is a misleading metric for wall-clock time if the communication can be hidden behind computation.

Let's first confirm the communication volume. Let $\Psi$ be the number of model parameters and $N_d$ be the number of devices.

*   **Standard DDP:** Performs one `all_reduce` on the gradients per step. An `all_reduce` is equivalent to a `reduce_scatter` followed by an `all_gather`. The total data sent and received by each GPU is approximately $2 \times |\Psi|$ bytes.
*   **FSDP (ZeRO-3):**
    1.  **Forward Pass:** One `all_gather` for each layer's parameters before its computation. Total over all layers: $|\Psi|$ bytes.
    2.  **Backward Pass:** One `all_gather` for each layer's parameters, followed by a `reduce_scatter` for its gradients. Total over all layers: $|\Psi| + |\Psi| = 2|\Psi|$ bytes.
    
    The total communication volume is approximately $3 \times |\Psi|$ bytes, which is indeed 1.5x that of standard DDP.

The key is that FSDP's communication is not a single blocking call at the end of the backward pass. Instead, it's broken into smaller, per-layer chunks that can be scheduled intelligently.

**Intuition: The FSDP Execution Timeline**

Modern GPU programming uses streams to execute operations asynchronously. We can have a `compute_stream` for calculations (like matrix multiplies) and a `comm_stream` for data transfers (like `all_gather`).

Consider the execution for two sequential layers, `Layer L` and `Layer L+1`, in an FSDP forward pass:

1.  **Initiate Comm for Layer L:** On the `comm_stream`, issue a non-blocking `all_gather` for the parameters of `Layer L`.
2.  **Wait and Compute Layer L:** The `compute_stream` waits for the `comm_stream` to finish gathering the parameters for `Layer L`. Once they arrive, it performs the forward computation for `Layer L`.
3.  **Initiate Comm for Layer L+1 & Free Layer L:** Crucially, *as soon as the computation for `Layer L` begins*, we can immediately issue the non-blocking `all_gather` for the parameters of `Layer L+1` on the `comm_stream`. After `Layer L`'s computation is done, its (now full-sized) parameters can be freed.
4.  **Wait and Compute Layer L+1:** The `compute_stream` finishes `Layer L`, then waits for the `all_gather` of `Layer L+1`'s parameters (which was already in flight) and computes `Layer L+1`.

This creates a pipelined effect:

```
Time ->
--------------------------------------------------------------------
Compute Stream: | Wait L | Compute L | Wait L+1 | Compute L+1 | ...
Comm Stream:    | Gather L | Gather L+1 | Gather L+2 | ...
--------------------------------------------------------------------
```

The communication for layer `L+1` is overlapped with the computation for layer `L`.

**Formal Condition for Perfect Overlap**

The communication latency is hidden if the time to compute is greater than or equal to the time to communicate.
$$ T_{\text{compute}}(L) \ge T_{\text{comm}}(L+1) $$

Let's model these times:

*   **Communication Time:** A common model for collective communication is $T_{\text{comm}}(M) = \alpha + \beta M$, where $\alpha$ is the latency overhead, $M$ is the message size in bytes, and $\beta$ is the reciprocal of the effective bandwidth. For an `all_gather` of a layer with $|\Psi_L|$ parameters (at 2 bytes/param), $M = 2|\Psi_L|$.
*   **Computation Time:** This is the number of floating-point operations (FLOPs) for the layer, divided by the GPU's effective TFLOPS. $T_{\text{compute}}(L) = \frac{\text{FLOPs}(L)}{\text{TFLOPS}_{\text{effective}}}$.

For a Transformer layer, the FLOPs are dominated by matrix multiplications and are proportional to `batch_size * seq_len * hidden_dim^2`. The parameters $|\Psi_L|$ are proportional to `hidden_dim^2`. For large hidden dimensions, the computation time grows faster than the communication time, making it easier to hide the latency.

**Pseudocode with Overlap**

```python
# Simplified FSDP forward pass with overlapping
compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

# Prefetch parameters for the first layer
with torch.cuda.stream(comm_stream):
    param_l = all_gather(param_shard_l) # Non-blocking

for l in range(num_layers):
    # Wait for the current layer's params to be ready
    compute_stream.wait_stream(comm_stream)

    with torch.cuda.stream(compute_stream):
        # Once params are ready, start computation for layer l
        activations = compute_forward(activations, param_l)
        # We no longer need the full params for layer l
        del param_l

    # As soon as computation for layer l starts, prefetch params for layer l+1
    if l < num_layers - 1:
        with torch.cuda.stream(comm_stream):
            param_l_plus_1 = all_gather(param_shard_l_plus_1) # Non-blocking

# Final synchronization
torch.cuda.synchronize()
```
The same logic applies to the backward pass, where the `reduce_scatter` of layer `L`'s gradients can be overlapped with the computation of layer `L-1`'s gradients.

**Pitfall:** This overlap is not guaranteed. If a model has many small layers (e.g., small `hidden_dim`), the `T_compute` for each layer might be too small to hide the `T_comm` (especially the latency component $\alpha$). In such cases, the GPU will spend time waiting for data, and FSDP's performance will degrade, exposing the higher communication volume as increased wall-clock time.

### Q: How does Megatron-style Tensor Parallelism handle a sequence of matrix multiplications, like in an MLP block, without requiring an expensive `all_gather` of the intermediate activation?

This is a clever algorithmic trick at the heart of tensor parallelism. It pairs a **column-parallel** linear layer with a subsequent **row-parallel** linear layer. This structure ensures the intermediate activation can remain sharded (distributed) across GPUs, avoiding a communication-heavy `all_gather` step.

Let's analyze a standard Transformer MLP block:
$$ Y = \text{GELU}(X W_{up}) W_{down} $$
Here, $X \in \mathbb{R}^{s \times h}$, $W_{up} \in \mathbb{R}^{h \times 4h}$, and $W_{down} \in \mathbb{R}^{4h \times h}$.

Let's parallelize this across $T$ GPUs.

**The Strategy: Column-Parallel followed by Row-Parallel**

1.  **Split `W_up` by columns:** Each GPU $i$ gets a shard $W_{up, i} \in \mathbb{R}^{h \times (4h/T)}$. This is a **column-parallel** split.
2.  **Split `W_down` by rows:** Each GPU $i$ gets a shard $W_{down, i} \in \mathbb{R}^{(4h/T) \times h}$. This is a **row-parallel** split.

**Forward Pass Derivation**

Let's assume the input $X$ is replicated on all $T$ GPUs (this is true if the preceding layer, e.g., attention, ended with an `all_reduce`).

1.  **First Matmul (Column-Parallel):** Each GPU $i$ computes its part of the first matrix multiplication:
    $$ Z_i = X W_{up, i} $$
    The result $Z_i$ has shape $s \times (4h/T)$. If we were to concatenate these results, we would get the full intermediate tensor $Z = [Z_1, Z_2, \dots, Z_T]$. But we don't! The tensor $Z$ remains sharded across the GPUs.

2.  **Activation Function:** The activation function (GELU) is applied element-wise on the local shard.
    $$ G_i = \text{GELU}(Z_i) $$
    The sharded tensor $G = [G_1, G_2, \dots, G_T]$ is still distributed. **No communication has occurred yet.**

3.  **Second Matmul (Row-Parallel):** The full mathematical operation is $Y = G W_{down}$. Let's expand this with our sharded tensors:
    $$ Y = [G_1, G_2, \dots, G_T] \begin{bmatrix} W_{down, 1} \\ W_{down, 2} \\ \vdots \\ W_{down, T} \end{bmatrix} = \sum_{i=1}^{T} G_i W_{down, i} $$
    This is a beautiful result. Each GPU $i$ can compute a partial result $Y_i = G_i W_{down, i}$ using only its local data. The final result $Y$ is simply the sum of these partial results.

4.  **Final Communication:** To get the final sum, we perform a single `all_reduce` operation on the partial results $Y_i$.
    $$ Y = \text{all_reduce}(\{Y_1, Y_2, \dots, Y_T\}) $$

The output $Y$ is now replicated on all GPUs, ready to be the input for the next Transformer block.

**Pseudocode of the Forward Pass**

```python
def mlp_forward(X, W_up_shard, W_down_shard, tp_group):
    # X is replicated on all GPUs in tp_group
    # W_up_shard is a column-wise shard of W_up
    # W_down_shard is a row-wise shard of W_down

    # 1. Column-parallel matmul. Output is sharded.
    # f_identity function in forward pass
    Z_shard = matmul(X, W_up_shard)

    # 2. Element-wise activation. Output remains sharded.
    G_shard = gelu(Z_shard)

    # 3. Row-parallel matmul. Output is a partial result.
    # g_all_reduce function in forward pass
    Y_partial = matmul(G_shard, W_down_shard)

    # 4. Sum partial results and replicate across GPUs.
    Y = all_reduce(Y_partial, group=tp_group)

    return Y
```

**Backward Pass**

The backward pass mirrors this logic. The gradient `dY` is replicated. The backward pass of an `all_reduce` is an identity operation, so each GPU starts with `dY`.

1.  The gradient for the row-parallel layer (`dG_shard`) will be sharded, which is exactly what's needed as input for the backward pass of the GELU and the column-parallel layer.
2.  The gradient with respect to the input, `dX`, will be a partial sum on each GPU, requiring a final `all_reduce` to be correctly accumulated.

This elegant pairing avoids materializing the full $s \times 4h$ intermediate tensor on any single GPU and collapses the communication into a single `all_reduce` at the end of the block.

### Q: In Pipeline Parallelism, increasing the number of microbatches `m` reduces the "bubble" overhead. However, very small microbatches can underutilize the GPU. How can we formally analyze this trade-off to find the optimal number of microbatches?

This is a classic systems trade-off between minimizing pipeline stalls and maximizing single-device performance. We can find the optimal number of microbatches, $m_{opt}$, by creating a simple cost model for the total execution time and minimizing it.

Let's define our variables:

*   $p$: Number of pipeline stages (GPUs).
*   $B$: Global batch size.
*   $m$: Number of microbatches.
*   $b$: Microbatch size, where $b = B/m$.

**1. Modeling Single-GPU Microbatch Time**

The time to process one microbatch of size $b$ on a single GPU, $T_{step}(b)$, is not perfectly linear. It includes fixed overheads (kernel launches, memory allocation) and a compute component. A simple but effective model is:
$$ T_{step}(b) = T_{ov} + T_{comp} \cdot b $$

*   $T_{ov}$: The fixed overhead per microbatch, independent of its size.
*   $T_{comp}$: The compute time per single example in the microbatch.

**2. Modeling Total Pipeline Execution Time**

Using the standard 1F1B (one forward, one backward) schedule, the pipeline has a "bubble" at the beginning (filling the pipe) and end (draining the pipe). The total time to process a global batch is the sum of the time for all microbatches to pass through the steady-state portion of the pipe, plus the bubble time.

The total duration consists of $(m + p - 1)$ "ticks," where each tick is the time to process one microbatch on one stage.
$$ T_{wall}(m) = (m + p - 1) \cdot T_{step}(b) $$
Note: This is a slight simplification. A more precise model for 1F1B would be `(m+p-1) * (T_f + T_b)`, but this simpler model captures the core trade-off.

Now, substitute our models for $T_{step}(b)$ and $b = B/m$:
$$ T_{wall}(m) = (m + p - 1) \left( T_{ov} + T_{comp} \frac{B}{m} \right) $$

Let's expand this expression:
$$ T_{wall}(m) = m T_{ov} + m T_{comp} \frac{B}{m} + (p-1)T_{ov} + (p-1)T_{comp}\frac{B}{m} $$
$$ T_{wall}(m) = m T_{ov} + T_{comp} B + (p-1)T_{ov} + \frac{(p-1)T_{comp}B}{m} $$

The term $T_{comp} B$ is the total compute time for the batch, which is constant. The term $(p-1)T_{ov}$ is also constant with respect to $m$. The parts that vary with $m$ are:

*   $m T_{ov}$: Total overhead cost. This increases linearly with $m$ (more, smaller microbatches means more overhead).
*   $\frac{(p-1)T_{comp}B}{m}$: Total bubble cost. This is the compute work of $(p-1)$ microbatches that is not parallelized. It decreases as $m$ increases.

**3. Finding the Optimal `m`**

To find the value of $m$ that minimizes $T_{wall}(m)$, we can take the derivative with respect to $m$ and set it to zero. We only need to consider the terms that depend on $m$.
$$ \frac{d}{dm} \left( m T_{ov} + \frac{(p-1)T_{comp}B}{m} \right) = 0 $$
$$ T_{ov} - \frac{(p-1)T_{comp}B}{m^2} = 0 $$
$$ T_{ov} = \frac{(p-1)T_{comp}B}{m^2} $$

Solving for $m$:
$$ m^2 = \frac{(p-1)B \cdot T_{comp}}{T_{ov}} $$
$$ m_{opt} = \sqrt{(p-1)B \frac{T_{comp}}{T_{ov}}} $$

**Interpretation of the Result**

This formula provides powerful intuition:

*   **Pipeline Depth (`p-1`):** A deeper pipeline (larger $p$) requires more microbatches to hide the larger bubble.
*   **Global Batch Size (`B`):** A larger global batch allows for more microbatches.
*   **Arithmetic Intensity (`T_comp / T_ov`):** This ratio is key.
    *   If computation is expensive relative to overhead ($T_{comp} \gg T_{ov}$), the ratio is large. It's worth it to have many small microbatches to reduce the bubble, as the overhead of each is negligible.
    *   If overhead is high relative to computation ($T_{ov} \gg T_{comp}$), e.g., for small layers or fast GPUs, the ratio is small. It's better to use fewer, larger microbatches to amortize the high fixed cost, even if it means accepting a larger pipeline bubble.

In practice, $T_{ov}$ and $T_{comp}$ must be empirically measured for a given model and hardware. However, this derivation proves that simply "making microbatches smaller to reduce the bubble" is not always the right answer. There is a sweet spot that optimally balances the two competing costs.