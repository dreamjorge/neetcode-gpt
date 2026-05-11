from typing import Dict, List, Tuple


class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        # --- WHAT IS A VOCABULARY? ---
        #
        # Every neural language model (GPT, LLaMA, Claude) works with NUMBERS,
        # not text. Before any learning can happen, we need a translation layer:
        #
        #   "hello" → [7, 4, 11, 11, 14]  → neural network → [7, 4, 11, 11, 14] → "hello"
        #              ↑ encode                                  ↑ decode
        #
        # A VOCABULARY is simply that two-way translation dictionary.
        # Here we build the simplest possible one: character-level.
        # Each unique character gets a unique integer ID.
        #
        # Real LLMs use BPE (Byte Pair Encoding) to build much larger
        # vocabularies of subword tokens — but the principle is identical.

        # --- STEP 1: Discover all unique characters in the text ---
        # `set(text)` removes duplicates.
        # `sorted(...)` orders them alphabetically → guarantees that the same
        # text always produces the same vocabulary (deterministic mapping).
        #
        # Example: text = "hello"
        #   set("hello")   → {'h', 'e', 'l', 'o'}   (order not guaranteed)
        #   sorted(...)    → ['e', 'h', 'l', 'o']    (always this order)
        unique_chars = sorted(set(text))

        # --- STEP 2: Build the string → integer mapping (encoder side) ---
        #
        # enumerate(['e','h','l','o']) yields: (0,'e'), (1,'h'), (2,'l'), (3,'o')
        # We flip the order to get  {'e':0, 'h':1, 'l':2, 'o':3}
        #
        # This is called `stoi`: String TO Integer
        char_to_id: Dict[str, int] = {char: idx for idx, char in enumerate(unique_chars)}

        # --- STEP 3: Build the integer → string mapping (decoder side) ---
        #
        # We invert char_to_id: swap keys and values.
        # {0:'e', 1:'h', 2:'l', 3:'o'}
        #
        # This is called `itos`: Integer TO String
        id_to_char: Dict[int, str] = {idx: char for char, idx in char_to_id.items()}

        return char_to_id, id_to_char

    def encode(self, text: str, char_to_id: Dict[str, int]) -> List[int]:
        # --- ENCODING: text → list of integer token IDs ---
        #
        # We replace each character with its corresponding integer ID
        # using the vocabulary we built in build_vocab.
        #
        # Example: text = "hello", char_to_id = {'e':0,'h':1,'l':2,'o':3}
        #
        #   'h' → 1
        #   'e' → 0
        #   'l' → 2
        #   'l' → 2
        #   'o' → 3
        #
        #   Result: [1, 0, 2, 2, 3]
        #
        # The neural network receives this list of integers — never raw text.
        return [char_to_id[char] for char in text]

    def decode(self, token_ids: List[int], id_to_char: Dict[int, str]) -> str:
        # --- DECODING: list of integer token IDs → text ---
        #
        # The reverse operation: map each integer back to its character,
        # then join them all into a single string.
        #
        # Example: token_ids = [1, 0, 2, 2, 3], id_to_char = {0:'e',1:'h',2:'l',3:'o'}
        #
        #   1 → 'h'
        #   0 → 'e'
        #   2 → 'l'
        #   2 → 'l'
        #   3 → 'o'
        #
        #   ''.join([...]) → "hello"  ✓
        #
        # This is how a language model converts its output numbers back into
        # readable text after generation.
        return ''.join(id_to_char[token_id] for token_id in token_ids)