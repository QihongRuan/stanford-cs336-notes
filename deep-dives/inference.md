## Inference Deep Dive: KV Cache, Batching, and Speculative Decoding

### Q: The lecture notes claim that during the token generation phase, the arithmetic intensity of attention is approximately 1 and that batching doesn't help, unlike for MLP layers. Can you provide a rigorous derivation for this and explain the core intuition behind why the batch size `B` cancels out?

This is a crucial observation that explains why the generation step of autoregressive decoding is fundamentally memory-bound. Let's break it down by analyzing the MLP and Attention layers separately.

**Recall:** Arithmetic Intensity (AI) is the ratio of floating-point operations (FLOPs) to bytes of data moved between high-bandwidth memory (HBM) and the processor.
$$ \text{AI} = \frac{\text{Total FLOPs}}{\text{Total Bytes Transferred}} $$
A workload is compute-bound if its AI is higher than the accelerator's intensity (e.g., ~295 for an H100); otherwise, it's memory-bound.

#### 1. MLP Layers during Generation

In the generation phase, we process a batch of `B` sequences, each generating one new token. So, the input to each MLP layer is a tensor of shape `(B, 1, D)`, where `D` is the model dimension. An MLP block typically consists of three matrix multiplications (up-projection, gate, down-projection). Let's analyze one, `Y = X @ W_up`, where `X` is `(B, D)` and `W_up` is `(D, F)`.

*   **FLOPs:** The number of operations for a matrix multiplication is `2 * B * D * F`.
*   **Bytes Transferred:**
    *   Read input `X`: `B * D * 2` bytes (assuming BF16).
    *   Read weights `W_up`: `D * F * 2` bytes.
    *   Write output `Y`: `B * F * 2` bytes.
    *   Total Bytes: `2 * (BD + DF + BF)`.

The arithmetic intensity is:
$$ \text{AI}_{\text{MLP}} = \frac{2 \cdot BDF}{2 \cdot (BD + DF + BF)} = \frac{BDF}{BD + DF + BF} $$

In a typical large model, the hidden dimensions `D` and `F` are much larger than the batch size `B` (e.g., `D, F` are in the thousands, `B` is tens or hundreds). Therefore, the `DF` term in the denominator dominates:
$$ \text{AI}_{\text{MLP}} \approx \frac{BDF}{DF} = B $$

**Intuition:** The largest piece of data moved is the weight matrix `W_up`, which is `O(DF)`. This single matrix is read from HBM and then re-used for all `B` sequences in the batch. The computation `O(BDF)` grows linearly with `B`, but the largest memory cost `O(DF)` is amortized. Thus, increasing the batch size `B` directly increases the arithmetic intensity, helping to make the operation compute-bound.

#### 2. Attention Layers during Generation

The situation is starkly different for attention. During generation, for each of the `B` sequences, the new query vector `q` (shape `(1, D)`) must attend to all `S` previous tokens in its own history, which are stored in its personal Key-Value (KV) cache.

Let's analyze the first major computation: the dot-product of queries with keys, `AttentionScores = Q @ K^T`.

*   `Q` is the batch of new queries: shape `(B, 1, D)`.
*   `K_{cache}` is the batch of cached keys: shape `(B, S, D)`.

*   **FLOPs:** We perform `B` independent matrix multiplications of a `(1, D)` matrix with a `(D, S)` matrix.
    *   Total FLOPs: `B \times (2 \cdot 1 \cdot S \cdot D) = 2 \cdot B \cdot S \cdot D`.
*   **Bytes Transferred:**
    *   Read queries `Q`: `B * 1 * D * 2` bytes.
    *   Read key cache `K_{cache}`: `B * S * D * 2` bytes.
    *   Write attention scores: `B * 1 * S * 2` bytes.
    *   Total Bytes (ignoring the small output): `2 \cdot (BD + BSD) = 2 \cdot BD(S+1)`.

The arithmetic intensity is:
$$ \text{AI}_{\text{Attention}} = \frac{2 \cdot BSD}{2 \cdot BD(S+1)} = \frac{S}{S+1} $$

As the context length `S` grows, this value approaches 1. Notice that the batch size `B` has completely canceled out.

**Intuition:** Unlike the MLP layers where a single large weight matrix is shared across the batch, in attention, **each sequence has its own private KV cache**. When you double the batch size `B`, you also double the total amount of KV cache data that must be read from HBM (`B * S * D`). Both the FLOPs and the memory traffic scale linearly with `B`, so their ratio—the arithmetic intensity—remains constant.

The same logic applies to the second major computation, `Output = AttentionScores @ V`, where the AI also works out to be independent of `B`.

**Conclusion:** Because the KV cache is not shared between sequences in a batch, batching does not amortize the memory access cost for attention layers during generation. The arithmetic intensity remains stuck at a very low value (`≈1`), far below the threshold for compute-bound execution on modern accelerators. This makes the generation step of Transformers inherently **memory-bandwidth bound**.

---

### Q: Speculative decoding accepts a token `x_i` from a draft model `p` with probability `min(1, q(x_i|H) / p(x_i|H))`, where `q` is the target model. How can we be sure this process generates a sequence with the exact same distribution as sampling directly from the target model `q`?

This is a fantastic question, as the correctness of speculative decoding relies on a clever application of rejection sampling. The guarantee that the final sample comes from the target distribution `q` is not immediately obvious. Let's derive it.

Let `H` be the history (prompt + previously generated tokens). We want to sample the next token `x` from the target distribution `q(x|H)`. We use a cheaper draft model that produces a distribution `p(x|H)`.

The algorithm for one step is as follows:

1.  Sample a candidate token `x' ~ p(x|H)`.
2.  Calculate the acceptance ratio `r = q(x'|H) / p(x'|H)`.
3.  Accept `x'` with probability `min(1, r)`.
4.  If `x'` is accepted, it becomes the final output for this step.
5.  If `x'` is rejected, we must sample a token from a modified "correction" distribution.

The key is to show that the combination of the accepted draft tokens and the corrected rejection samples perfectly reconstructs the original target distribution `q`.

#### Derivation

Let's find the probability of sampling a specific token `x_k` as the final output, `P(x_{final} = x_k)`. A token can be chosen in two mutually exclusive ways: either it was the draft token and was accepted, or the draft token was rejected and it was chosen via the correction mechanism.

**Case 1: The draft token `x_k` is sampled and accepted.**

The probability of this joint event is:
$$ P(\text{sample } x_k \text{ and accept } x_k) = p(x_k|H) \times \min\left(1, \frac{q(x_k|H)}{p(x_k|H)}\right) $$
This simplifies to:
$$ P(\text{sample } x_k \text{ and accept } x_k) = \min(p(x_k|H), q(x_k|H)) $$

This is the portion of the probability mass for `x_k` that is "covered" by the acceptance step. However, this is clearly not equal to `q(x_k|H)`. The missing mass must be accounted for by the rejection case.

**Case 2: The draft token is rejected, and `x_k` is sampled from the correction distribution.**

First, let's define the "missing" probability mass for each token `x_k`. This is the probability that `q` assigns to `x_k` minus the probability that our acceptance step assigns to it.
$$ \text{MissingMass}(x_k) = q(x_k|H) - \min(p(x_k|H), q(x_k|H)) $$
This can be written more cleanly using the positive part function `(z)^+ = max(0, z)`:
$$ \text{MissingMass}(x_k) = (q(x_k|H) - p(x_k|H))^+ $$
If `q(x_k|H) <= p(x_k|H)`, the missing mass is 0. If `q(x_k|H) > p(x_k|H)`, the acceptance step only covered `p(x_k|H)`, so we are missing `q(x_k|H) - p(x_k|H)`.

The total probability of rejecting the draft sample is the sum of all missing masses over the entire vocabulary `V`:
$$ P(\text{reject}) = \sum_{x \in V} \text{MissingMass}(x) = \sum_{x \in V} (q(x|H) - p(x|H))^+ $$

If a rejection occurs, we must sample from the normalized distribution of this missing mass. This is our **correction distribution**, `q_{corr}`:
$$ q_{corr}(x_k|H) = \frac{\text{MissingMass}(x_k)}{P(\text{reject})} = \frac{(q(x_k|H) - p(x_k|H))^+}{\sum_{x \in V} (q(x|H) - p(x|H))^+} $$

The probability of sampling `x_k` via this rejection path is:
$$ P(\text{reject and then sample } x_k) = P(\text{reject}) \times q_{corr}(x_k|H) $$
$$ P(\text{reject and then sample } x_k) = P(\text{reject}) \times \frac{(q(x_k|H) - p(x_k|H))^+}{P(\text{reject})} = (q(x_k|H) - p(x_k|H))^+ $$

**Putting It All Together**

The total probability of `x_k` being the final sample is the sum of the probabilities from the two cases:
$$ P(x_{final} = x_k) = P(\text{sample } x_k \text{ and accept } x_k) + P(\text{reject and then sample } x_k) $$
$$ P(x_{final} = x_k) = \min(p(x_k|H), q(x_k|H)) + (q(x_k|H) - p(x_k|H))^+ $$

Let's check the two possibilities for the relationship between `q` and `p`:

1.  If `q(x_k|H) \le p(x_k|H)`:
    $P(x_{final} = x_k) = q(x_k|H) + 0 = q(x_k|H)$.
2.  If `q(x_k|H) > p(x_k|H)`:
    $P(x_{final} = x_k) = p(x_k|H) + (q(x_k|H) - p(x_k|H)) = q(x_k|H)$.

In both cases, the final probability is exactly `q(x_k|H)`. The procedure is mathematically guaranteed to be an exact sample from the target model `q`.

```python
def speculative_sample_step(history, target_model, draft_model):
    """
    Performs one step of speculative sampling.
    Note: In practice, this is done for a block of K tokens.
    """
    # 1. Draft: Sample a candidate from the cheap model p
    draft_logits = draft_model.get_logits(history)
    draft_probs = softmax(draft_logits)
    candidate_token = sample_from(draft_probs)

    # 2. Verify: Get probabilities for the candidate from both models
    # This is the expensive step, but it's just one forward pass
    target_logits = target_model.get_logits(history)
    target_probs = softmax(target_logits)

    p_prob = draft_probs[candidate_token]
    q_prob = target_probs[candidate_token]

    # 3. Accept/Reject
    acceptance_prob = min(1.0, q_prob / p_prob)
    if random.random() < acceptance_prob:
        # Accept the draft token
        return candidate_token
    else:
        # 4. Reject and Resample from the correction distribution
        # The correction distribution is proportional to (q - p)^+
        # This can be implemented efficiently without re-calculating softmax
        correction_probs = (target_probs - draft_probs).clamp(min=0)
        norm_factor = correction_probs.sum()
        if norm_factor < 1e-9: # Handle case where p >= q everywhere
            # This can happen if p is a perfect or overly confident model
            # In this rare case, we must resample from q.
            return sample_from(target_probs)
        
        correction_probs /= norm_factor
        return sample_from(correction_probs)

```
**Intuition:** The draft model `p` acts as a cheap proposal distribution. We use it to quickly guess a token. If the target model `q` would have been *at least as likely* to pick that token (`q/p >= 1`), we just accept it. If `q` was less likely (`q/p < 1`), we accept it some of the time. The magic is in the rejection step: we precisely calculate the "probability mass" that our acceptance rule missed and sample from that residual. This perfectly patches the distribution back up to match `q`.

---

### Q: PagedAttention is said to use a "copy-on-write" mechanism to efficiently handle multiple output sequences sharing a common prefix. How is this implemented at the level of KV cache memory blocks, and what is the precise memory saving compared to a naive implementation?

This question targets the core systems innovation of PagedAttention, which is crucial for high-throughput serving. The "copy-on-write" analogy from operating systems is powerful but can be slightly misleading; for standard decoding, it's more of a "share-then-append" mechanism.

#### The Problem: Shared Prefixes in Inference

A common inference scenario involves generating multiple sequences from a single prompt. Examples include:

*   **Beam Search:** Generating `k` candidate sequences (beams).
*   **Parallel Sampling:** Generating `N` independent samples for one user query to find the best one.
*   **Batching identical prompts:** Multiple users asking the same question.

In all these cases, the initial processing of the prompt (the "prefill" stage) results in an identical KV cache for all `N` sequences.

#### Naive KV Cache Management

Without PagedAttention, an inference server would typically allocate a contiguous block of memory for the KV cache of each sequence, large enough to hold the prompt and the maximum possible generation length.

*   **Implementation:** After running prefill on the prompt once, the resulting KV cache data would be physically copied (`memcpy`) into the `N` separate buffers.
*   **Memory Cost:** Let `S_p` be the prompt length, `L` be the number of layers, and `D` be the model dimension. The memory for one token's K and V vectors (in MHA) is `2 * L * D * sizeof(dtype)`. Let this be `C_token`. The total memory for the `N` prefixes is:
    $$ M_{\text{naive}} = N \times S_p \times C_{\text{token}} $$
This is extremely wasteful, as `(N-1) * S_p * C_token` bytes are redundant copies. This wasted memory reduces the number of concurrent requests a GPU can handle.

#### PagedAttention: Virtualizing the KV Cache

PagedAttention solves this by borrowing the concept of virtual memory and paging from operating systems.

1.  **Physical Blocks:** The entire GPU memory allocated for the KV cache is divided into fixed-size blocks (or "pages").
2.  **Logical Blocks & Page Tables:** Each sequence's KV cache is a *logical* sequence of blocks. A per-sequence **page table** maps these logical block indices to physical block indices in GPU memory. This means a sequence's KV cache can be stored in non-contiguous physical blocks.
3.  **Reference Counting:** Each physical block maintains a reference counter.

**"Copy-on-Write" in Action (Share-then-Append):**

Let's trace the `N` parallel samples scenario:

1.  **Prefill:** The prompt is processed once. The resulting KV cache occupies, say, `k` physical blocks.
2.  **Sharing:** Instead of copying, PagedAttention creates `N` logical sequences. The page tables for all `N` sequences are initialized to point to the *same* `k` physical blocks. The reference count of each of these `k` blocks is set to `N`.
3.  **Generation (Divergence):** Now, we generate the first new token for each of the `N` sequences.
    *   For sequence `i`, the model generates a new key and value vector.
    *   The PagedAttention memory manager allocates a **new, free physical block** from its pool.
    -   The page table for sequence `i` is appended with the index of this new physical block.
    -   The reference count of this new block is set to 1.
    *   The reference counts of the original `k` shared blocks remain unchanged at `N`.

No data from the prefix was ever copied. The sequences continue to share the read-only prefix blocks while appending their own unique, private blocks for the newly generated tokens.

```python
# Conceptual pseudocode for PagedAttention's sharing mechanism
class MemoryManager:
    # Manages a pool of physical blocks and their reference counts
    physical_blocks: dict[int, KVCacheBlock]
    ref_counts: dict[int, int]
    free_block_indices: list[int]

    def allocate_block(self) -> int:
        # ... returns index of a free block
    
    def free_block(self, block_idx: int):
        # ... decrements ref_count and adds to free pool if zero

class Sequence:
    # Each sequence has a page table mapping logical to physical blocks
    page_table: list[int] # list of physical block indices

# --- Scenario: N=3 parallel samples from one prompt ---

manager = MemoryManager()

# 1. Prefill: Process prompt, store in new physical blocks
prompt_kv_data = prefill(prompt)
prompt_physical_blocks = manager.store_in_new_blocks(prompt_kv_data) # e.g., [101, 102]

# 2. Share: Create 3 sequences pointing to the same physical blocks
sequences = [Sequence() for _ in range(3)]
for seq in sequences:
    seq.page_table = list(prompt_physical_blocks) # All share [101, 102]

# Set reference counts for the shared blocks
for block_idx in prompt_physical_blocks:
    manager.ref_counts[block_idx] = 3

# 3. Generate (Diverge): Generate one token for each sequence
for i, seq in enumerate(sequences):
    # Generate new KV for this sequence's next token
    new_kv_data = generate_step(seq) 
    
    # Allocate a new physical block for this unique token
    new_block_idx = manager.allocate_block() # e.g., 201 for seq 0, 202 for seq 1, etc.
    manager.store_data(new_block_idx, new_kv_data)
    manager.ref_counts[new_block_idx] = 1
    
    # Append the new physical block to this sequence's page table
    seq.page_table.append(new_block_idx)

# After one step:
# seq[0].page_table = [101, 102, 201]
# seq[1].page_table = [101, 102, 202]
# seq[2].page_table = [101, 102, 203]
# manager.ref_counts = {101: 3, 102: 3, 201: 1, 202: 1, 203: 1}
```

#### Memory Savings Analysis

Let `S_p` be the prompt length, `S_g` be the generated length per sequence, and `C_token` be the memory cost per token in the KV cache.

*   **Naive Memory:** Each of the `N` sequences requires its own buffer for the full length.
    $$ M_{\text{naive}} = N \times (S_p + S_g) \times C_{\text{token}} $$
*   **PagedAttention Memory:** The prefix is stored once, and each of the `N` sequences gets its own memory for the generated part.
    $$ M_{\text{PagedAttention}} = (S_p \times C_{\text{token}}) + (N \times S_g \times C_{\text{token}}) $$
*   **Memory Saved:** The difference is the memory saved by not duplicating the prefix.
    $$ \text{Savings} = M_{\text{naive}} - M_{\text{PagedAttention}} = (N-1) \times S_p \times C_{\text{token}} $$

This saving is immense, especially for applications with long system prompts or few-shot examples (`S_p` is large) and a high degree of parallelism (`N` is large). By eliminating this redundancy, PagedAttention can dramatically increase the effective batch size and overall throughput of an inference server.