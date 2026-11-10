## Tokenization & Byte-Pair Encoding

### Q: A naive implementation of BPE training involves repeatedly scanning the entire corpus to find the most frequent pair, which has a terrible time complexity of roughly $O(\text{CorpusSize} \times \text{NumMerges})$. How is BPE training implemented efficiently in practice?

A thorough and efficient implementation of the BPE training algorithm is critical for handling large corpora. The naive approach is computationally infeasible. The key to an efficient implementation lies in using appropriate data structures to avoid re-scanning the corpus for every single merge.

The core operations in each step of the training loop are:

1.  Find the most frequent adjacent pair of tokens.
2.  Merge this pair into a new token.
3.  Update the corpus and pair counts for the next iteration.

An efficient algorithm optimizes these steps as follows:

**1. Initial Corpus Representation and Pair Counting**

Instead of treating the corpus as a monolithic flat array of bytes, we first process it into a more structured format. A common approach is to split the text into "words" (e.g., by whitespace) and represent each word as a sequence of tokens. This is not strictly necessary for the algorithm's efficiency but aligns with common practice (like in GPT-2).

The most important step is to perform **a single pass** over the entire corpus at the beginning to build a complete frequency count of all adjacent token pairs. This can be stored in a hash map.

```python
# Pseudocode for initial counting
def get_initial_pair_counts(corpus_words):
    pair_counts = {}
    for word in corpus_words:
        tokens = list(word.encode('utf-8')) # or some other initial tokenization
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts
```
This initial counting step has a complexity of $O(C)$, where $C$ is the size of the corpus.

**2. Efficiently Finding the Best Pair**

With the `pair_counts` map, finding the most frequent pair is a simple iteration over the map's values. We don't need a priority queue for this, as we can just find the max in each step.

**3. Efficiently Updating Counts After a Merge**

This is the most complex part. When we merge a pair, say `(A, B)` into a new token `Z`, we cannot afford to rescan the corpus to update counts. The merge only affects the local context. Consider a sequence `... X, A, B, Y ...`.

- The pair `(A, B)` is eliminated.
- The pairs `(X, A)` and `(B, Y)` are destroyed.
- New pairs `(X, Z)` and `(Z, Y)` are created.

To perform this update efficiently, we need to know where all occurrences of `(A, B)` are. A naive search is too slow. The efficient solution is to maintain an **inverted index**: a map from each pair to a list of locations (e.g., word index and position within the word) where it occurs.

However, maintaining a full inverted index is complex. A more practical and widely used approach (e.g., in `huggingface/tokenizers`) is to represent each word as a **doubly linked list** of tokens. This allows for $O(1)$ modification of the sequence during a merge.

The refined algorithm looks like this:

1.  **Initialization:**
    -   Represent the corpus as a list of "words," where each word is a doubly linked list of its initial tokens (bytes).
    -   Perform one pass to compute initial `pair_counts`.

2.  **Merge Loop:** For `i` from 1 to `num_merges`:

    1. **Find Best Pair:** Find the pair `(A, B)` with the highest count in `pair_counts`.

    2. **Update Corpus and Counts:** Iterate through each word in the corpus. Within each word (linked list), iterate through its nodes.

        - If a node and its successor form the pair `(A, B)`:

            1. Let the preceding node be `X` and the succeeding node be `Y`.

            2. **Decrement old counts:**

                - `pair_counts[(X.value, A)] -= 1`
                - `pair_counts[(B, Y.value)] -= 1`

            3. **Merge in the linked list:** Change the current node's value to the new token `Z` and delete the next node (which held `B`). This is an $O(1)$ pointer operation.

            4. **Increment new counts:**

                - `pair_counts[(X.value, Z)] += 1`
                - `pair_counts[(Z, Y.value)] += 1`

    3. **Store Merge:** Add the merge rule `(A, B) -> Z` to our list of learned merges.

**Complexity Analysis:**

-   **Initial Count:** $O(C)$
-   **Merge Loop:** The loop runs $V - V_0$ times, where $V$ is the final vocabulary size and $V_0$ is the initial one (256).
-   Inside the loop, finding the best pair takes $O(P_i)$, where $P_i$ is the number of unique pairs at step $i$.
-   The crucial update step still requires iterating through the locations of the pair being merged. While we don't rescan the *entire* corpus, we do have to find all occurrences of the pair `(A, B)`. The linked list structure makes the *local modification* $O(1)$, but we still need to find where to perform the modification.

The key insight is that we can update the `pair_counts` dictionary directly without re-scanning the raw text. The cost of each merge step is proportional to the number of occurrences of the pair being merged, which is much smaller than the total corpus size. This makes the overall process tractable, though still computationally intensive for large vocabularies and corpora. The final complexity is much better than the naive $O(C \times V)$, closer to $O(C + \sum_{i=1}^{V-V_0} \text{occurrences}(\text{pair}_i) \log P_i)$.

---

### Q: The BPE encoding process involves applying a learned list of merges greedily. What happens if a string contains multiple, overlapping candidate pairs? For example, if our vocabulary has merges for `('a', 'b')` -> `'ab'` and `('b', 'c')` -> `'bc'`, how is the string `'abc'` tokenized?

This is an excellent question that probes a subtle but critical aspect of the BPE algorithm: ensuring deterministic encoding. If the string `'abc'` could be tokenized as either `['ab', 'c']` or `['a', 'bc']`, the tokenizer would be ambiguous and unusable.

The standard BPE encoding algorithm resolves this ambiguity with a simple, deterministic, greedy procedure. The key is that merges are not chosen based on what's possible, but are ranked by the order in which they were learned during training.

**The Algorithm for Deterministic Encoding:**

1.  **Initialization:** Convert the input string into an initial sequence of tokens (e.g., UTF-8 bytes).
    `'abc'` -> `[97, 98, 99]`

2.  **Iterative Merging:** Repeatedly scan the *current* token sequence and perform the *single best possible merge*. The "best" merge is defined as the one whose pair has the **lowest rank** (i.e., was learned earliest during training). If there's a tie in rank (which shouldn't happen with a proper merge list) or multiple occurrences of the same highest-priority pair, the **leftmost occurrence** is chosen.

Let's trace the example. Assume our training process learned the following merges in order:

-   ...
-   Merge #50: `(97, 98)` -> `256` (representing `'ab'`)
-   ...
-   Merge #100: `(98, 99)` -> `257` (representing `'bc'`)
-   ...

Here, the pair `(97, 98)` has a lower rank (50) than `(98, 99)` (100).

**Encoding `'abc'`:**

-   **Initial tokens:** `[97, 98, 99]`
-   **Iteration 1:**
    -   Scan the sequence for possible merges. We find two: `(97, 98)` at index 0 (rank 50) and `(98, 99)` at index 1 (rank 100).
    -   The best pair is `(97, 98)` because it has the lower rank (50 < 100).
    -   Perform this merge: replace `[97, 98]` with the new token `256`.
    -   The sequence becomes: `[256, 99]`.

-   **Iteration 2:**
    -   Scan the new sequence `[256, 99]`.
    -   The only adjacent pair is `(256, 99)`. Let's assume this pair was never learned as a merge during training.
    -   Since no more merges from our learned rules can be applied, the process terminates.

-   **Final Tokenization:** `[256, 99]`

The result is unambiguous. The greedy, rank-based, left-to-right priority ensures that for any given string and set of merge rules, there is only one possible output.

**Pseudocode for Encoding:**

```python
def get_pairs(tokens):
    """Helper to get all adjacent pairs in a sequence."""
    return set(zip(tokens, tokens[1:]))

def encode_bpe(text, merge_ranks):
    """
    Encodes text using BPE.
    merge_ranks: A dict mapping (tok1, tok2) -> rank (int, lower is better)
    """
    if not text:
        return []
    
    tokens = list(text.encode('utf-8'))

    while True:
        pairs = get_pairs(tokens)
        if not pairs:
            break

        # Find the best pair to merge based on rank
        best_pair = min(pairs, key=lambda p: merge_ranks.get(p, float('inf')))

        # If no mergeable pairs are found, we're done
        if best_pair not in merge_ranks:
            break

        # Find the first occurrence of the best pair and merge it
        first_idx = -1
        for i in range(len(tokens) - 1):
            if (tokens[i], tokens[i+1]) == best_pair:
                first_idx = i
                break
        
        # Perform the merge
        new_token_id = merge_ranks[best_pair] # In a real implementation, you'd map pair to new ID
        tokens = tokens[:first_idx] + [new_token_id] + tokens[first_idx+2:]

    return tokens
```

**Common Pitfall:** A frequent misconception is to iterate through the list of merge rules and apply each one to the entire string. This is incorrect and would lead to ambiguity. The correct method is to repeatedly find the single highest-priority merge *in the current state of the token sequence* and apply only that one before re-evaluating.

---

### Q: The lecture notes mention that GPT-2's BPE doesn't run on the raw byte stream of the entire corpus, but first splits text into "words" and runs BPE *within* each word. Why add this extra pre-tokenization step, and what are its algorithmic and practical consequences?

This design choice in the original GPT-2 tokenizer is a crucial heuristic that significantly shapes the nature of the resulting vocabulary and the tokenizer's behavior. It represents a trade-off between a "pure" statistical approach and a linguistically-motivated one.

**Motivation: Preventing "Meaningless" Cross-Word Merges**

The primary reason for this pre-tokenization step is to preserve the integrity of words as fundamental semantic units. A pure byte-level BPE, operating on a continuous stream of text, would frequently find statistically common pairs that cross word boundaries.

For example, in the text "this is a sentence.", the pair `('s', ' ')` (from "this is") might be very common. Merging this would create a token `s ` that combines the end of one word with a space. While statistically valid, this token is semantically messy. It conflates morphology (the plural 's') with syntax (word separation).

By first splitting the text into "word-like" chunks (e.g., using a regular expression that splits on whitespace and punctuation) and running BPE *only within these chunks*, the algorithm is constrained. It can learn to merge `t` and `h` to `th`, and `th` and `e` to `the`, but it can never merge the `e` from `the` with the space that follows it.

**Algorithmic and Practical Consequences:**

This design has profound consequences, creating both benefits and drawbacks.

**1. Pro: A "Cleaner," More Interpretable Vocabulary**
The resulting vocabulary is more intuitive. It consists of:

-   Complete common words (e.g., `" the"`, `" and"`, `" in"`).
-   Common sub-words and morphemes (e.g., `"ing"`, `"ation"`, `" pre"`).
This arguably creates a better inductive bias for the language model, as the tokens it operates on correspond more closely to semantic building blocks.

**2. Con: The "Leading Space" Problem**
This is the most significant drawback. Because pre-tokenization splits on whitespace, a word at the beginning of a sentence is treated as a different initial chunk from the same word in the middle of a sentence.

-   `"Tokenization is tricky."` -> `["Tokenization", " is", " tricky", "."]`
-   `" is"` and `"is"` are different strings and will be processed independently by BPE.

This means `" world"` and `"world"` will likely have different token IDs, effectively doubling the vocabulary required for many words. This is inefficient and can be a source of subtle bugs when manipulating text for a model. For example, `model.generate("Hello" + " world")` might produce a different result from `model.generate("Hello world")`.

**3. Con: Dependence on Heuristic Regex**
The entire tokenization process becomes dependent on the specific regular expression used for the initial split. The GPT-2 regex is famously complex:
```regex
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```
This pattern hard-codes rules about English contractions and how to group letters, numbers, and symbols. This introduces a strong, language-specific bias and makes the tokenizer less universal. A different regex would produce a completely different vocabulary and tokenization scheme.

**4. The Modern Approach: A More Principled Handling of Whitespace**
Recognizing these issues, many modern tokenizers like **SentencePiece** (used in Llama, T5) and the `huggingface/tokenizers` library take a more elegant approach. Instead of using pre-tokenization to prevent cross-word merges, they treat whitespace as a normal part of the sequence.

-   **The Meta-Character Trick:** Whitespace is often replaced with a special meta-character, like ` ` (U+2581), before BPE training. The string `"Hello world"` becomes `"Hello world"`.
-   Now, BPE can operate on the full sequence. It might learn to merge `" w"`, `"o"`, `"r"`, `"l"`, `"d"` into a single ` world` token. It is free to merge across the ` ` character if it's statistically frequent, but the space information is preserved within the token itself.

This avoids the "leading space" problem (a word is always preceded by a ` ` or is at the start of the sequence) and eliminates the need for a complex, heuristic pre-tokenization regex, resulting in a more robust and consistent system.