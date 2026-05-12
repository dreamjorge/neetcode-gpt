from typing import List, Dict


class Solution:

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # --- WHAT IS NUMBER TOKENIZATION? ---
        #
        # LLMs struggle with arithmetic partly because numbers get split
        # into tokens in unexpected ways. Understanding HOW a number gets
        # tokenized helps explain model behavior.
        #
        # Example with a typical BPE vocab:
        #   "42"    → ["42"]          (common number, one token)
        #   "1337"  → ["13", "37"]    (split at a vocab boundary)
        #   "99999" → ["999", "99"]   (greedy match from left)
        #
        # This function converts each integer to its string representation
        # and then tokenizes it using the shared greedy algorithm below.

        tokenized_numbers: List[List[str]] = []

        for number in numbers:
            # Convert the integer to its string form first.
            # 1024 → "1024"
            number_as_text = str(number)

            # Tokenize the string using the vocabulary.
            tokens = self._greedy_tokenize(number_as_text, vocab)
            tokenized_numbers.append(tokens)

        return tokenized_numbers

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # --- TOKEN COUNT ---
        #
        # Returns how many tokens a given text produces.
        # This directly affects cost and context window usage in real LLMs:
        # OpenAI charges per token, and every model has a max token limit.
        #
        # Example:
        #   "hello world" with a character vocab → 11 tokens (each char)
        #   "hello world" with a BPE vocab       →  2 tokens ("hello", " world")
        tokens = self._greedy_tokenize(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # --- FERTILITY SCORE ---
        #
        # Fertility = tokens / words
        # It measures how "expensive" a text is to represent in a given vocabulary.
        #
        #   fertility = 1.0  → every word maps to exactly one token (ideal)
        #   fertility > 1.0  → words are being split into multiple tokens
        #   fertility < 1.0  → multiple words are merged into one token (rare)
        #
        # Example:
        #   text = "hello world"
        #   tokens = ["hel", "lo", "world"]   → 3 tokens
        #   words  = ["hello", "world"]        → 2 words
        #   fertility = 3 / 2 = 1.5
        #
        # Real-world insight: languages with long compound words (German, Finnish)
        # tend to have higher fertility scores with English-trained vocabularies,
        # meaning they consume more tokens per word → higher API costs.
        tokens = self._greedy_tokenize(text, vocab)
        words  = text.split()
        return round(len(tokens) / len(words), 4)

    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        # --- STRATEGY: Greedy Longest-Match Tokenization ---
        #
        # This is the core algorithm used by most BPE tokenizers (GPT, LLaMA).
        #
        # Rule: at each position, consume the LONGEST substring that exists
        # in the vocabulary. If nothing matches, take just the single character
        # (fallback — ensures we never get stuck).
        #
        # "Greedy" means we commit to the longest match immediately,
        # without backtracking to explore shorter alternatives.
        #
        # Visual example:
        #   text  = "highest"
        #   vocab = {"high": 1, "hi": 2, "est": 3, "h": 4, ...}
        #
        #   i=0: try "highest"→no, "highes"→no, "highe"→no, "high"→YES
        #        → append "high", jump to i=4
        #   i=4: try "est"→YES
        #        → append "est", jump to i=7
        #   Result: ["high", "est"]

        tokens: List[str] = []
        position = 0

        while position < len(text):

            best_match: str | None = None

            # Try substrings from longest to shortest starting at `position`.
            # `range(len(text) - position, 0, -1)` counts down from the
            # maximum possible length to 1, ensuring we always try the
            # longest match first.
            for length in range(len(text) - position, 0, -1):
                substring = text[position : position + length]

                if substring in vocab:
                    best_match = substring
                    break   # stop as soon as we find the longest match

            if best_match is None:
                # --- FALLBACK: unknown character ---
                # No substring starting here exists in the vocab.
                # Consume just the single character so we always make progress.
                # This prevents an infinite loop on unseen characters.
                tokens.append(text[position])
                position += 1
            else:
                # --- MATCH FOUND: consume it and advance ---
                tokens.append(best_match)
                position += len(best_match)

        return tokens