from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # --- STRATEGY: Byte Pair Encoding (BPE) ---
        #
        # BPE is the tokenization algorithm used by GPT, LLaMA, and most
        # modern Large Language Models. It answers the question:
        #   "How should we split text into tokens for a neural network?"
        #
        # Core idea: start with individual characters, then repeatedly
        # MERGE the most frequent adjacent pair into a single new token.
        #
        # Example with corpus = "aabdaabd":
        #
        #   Start:    ['a','a','b','d','a','a','b','d']
        #   Step 1:   'a'+'a' appears 2x → merge → ['aa','b','d','aa','b','d']
        #   Step 2:   'aa'+'b' appears 2x → merge → ['aab','d','aab','d']
        #   Step 3:   'aab'+'d' appears 2x → merge → ['aabd','aabd']
        #
        # Each merge record tells the model: "these two tokens fuse into one."
        # The list of merges IS the vocabulary-building rulebook.

        # Split the corpus into individual characters — the starting vocabulary.
        # Every character is its own token at the beginning.
        tokens = list(corpus)

        # `merge_rules` records each merge in order: [[a, b], [aa, b], ...]
        # This ordered list is what the function returns.
        merge_rules = []

        for _ in range(num_merges):

            # Safety: we need at least 2 tokens to form any pair.
            if len(tokens) < 2:
                break

            # --- STEP 1: Count the frequency of every adjacent pair ---
            #
            # We scan the token list left to right, looking at every
            # consecutive (tokens[i], tokens[i+1]) pair.
            #
            # Example: tokens = ['a','a','b','d','a','a','b','d']
            #   pair ('a','a') → appears at index 0 and 4 → count = 2
            #   pair ('a','b') → appears at index 1 and 5 → count = 2
            #   pair ('b','d') → appears at index 2 and 6 → count = 2
            pair_frequencies = {}
            for i in range(len(tokens) - 1):
                adjacent_pair = (tokens[i], tokens[i + 1])
                pair_frequencies[adjacent_pair] = pair_frequencies.get(adjacent_pair, 0) + 1

            if not pair_frequencies:
                break

            # --- STEP 2: Select the best pair to merge ---
            #
            # "Best" = most frequent.
            # Tiebreak rule: if multiple pairs share the highest count,
            # pick the lexicographically smallest one (alphabetical order).
            # This ensures deterministic, reproducible results.
            #
            # Example: if ('a','b') and ('b','d') both appear 2 times:
            #   sorted(['ab', 'bd']) → 'ab' comes first → pick ('a','b')
            highest_frequency = max(pair_frequencies.values())

            # Collect all pairs that share the highest frequency, then sort
            # them alphabetically to apply the tiebreak rule consistently.
            top_pairs = sorted(
                pair for pair, count in pair_frequencies.items()
                if count == highest_frequency
            )
            best_pair = top_pairs[0]

            # Record this merge rule: [left_token, right_token]
            merge_rules.append([best_pair[0], best_pair[1]])

            # --- STEP 3: Apply the merge to the token list ---
            #
            # Scan left to right. Whenever we see the best_pair side by side,
            # replace both tokens with their concatenation.
            # We skip OVERLAPPING matches (i += 2 jumps past the merged token).
            #
            # Example: best_pair = ('a','a'), tokens = ['a','a','a','b']
            #   i=0: ('a','a') matches → append 'aa', skip to i=2
            #   i=2: ('a','b') no match → append 'a', move to i=3
            #   i=3: append 'b'
            #   Result: ['aa', 'a', 'b']   ← non-overlapping, left to right
            merged_tokens = []
            i = 0
            while i < len(tokens):
                # Check if current and next token form the best pair
                is_not_last      = i < len(tokens) - 1
                left_matches     = tokens[i]     == best_pair[0]
                right_matches    = tokens[i + 1] == best_pair[1] if is_not_last else False

                if is_not_last and left_matches and right_matches:
                    # Fuse both tokens into one and skip over them
                    merged_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    # No match — keep the token as-is
                    merged_tokens.append(tokens[i])
                    i += 1

            tokens = merged_tokens

        return merge_rules