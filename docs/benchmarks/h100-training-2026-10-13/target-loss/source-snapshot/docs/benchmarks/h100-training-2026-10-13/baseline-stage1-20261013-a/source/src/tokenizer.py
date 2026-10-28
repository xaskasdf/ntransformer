"""
Tokenizer wrapper for TinyLlama.

Supports:
- Custom BPE tokenizer (1024 tokens for 226K model)
- Llama 2 tokenizer (32K tokens for 110M model)
- ChatML special tokens
"""

import os
from typing import List, Optional, Union
from pathlib import Path

import sentencepiece as spm


# ChatML special tokens
SPECIAL_TOKENS = {
    'im_start': '<|im_start|>',
    'im_end': '<|im_end|>',
    'system': '<|system|>',
    'user': '<|user|>',
    'assistant': '<|assistant|>',
    'pad': '<|pad|>',
    'unk': '<|unk|>',
    'bos': '<|bos|>',
    'eos': '<|eos|>',
}


class Tokenizer:
    """Tokenizer wrapper for SentencePiece with ChatML support."""

    def __init__(self, model_path: str):
        """
        Initialize tokenizer from SentencePiece model.

        Args:
            model_path: Path to .model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        # Get vocab size
        self._vocab_size = self.sp.GetPieceSize()

        # Map special tokens to IDs
        self._special_token_ids = {}
        for name, token in SPECIAL_TOKENS.items():
            token_id = self.sp.PieceToId(token)
            if token_id != self.sp.unk_id():
                self._special_token_ids[name] = token_id

        # Convenience attributes
        self.bos_id = self._special_token_ids.get('bos', self.sp.bos_id())
        self.eos_id = self._special_token_ids.get('eos', self.sp.eos_id())
        self.pad_id = self._special_token_ids.get('pad', self.sp.pad_id())
        self.unk_id = self.sp.unk_id()

        self.im_start_id = self._special_token_ids.get('im_start')
        self.im_end_id = self._special_token_ids.get('im_end')

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token

        Returns:
            List of token IDs
        """
        tokens = self.sp.Encode(text)

        if add_bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if add_eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]

        return tokens

    def decode(self, tokens: List[int], skip_special: bool = False) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs
            skip_special: Skip special tokens in output

        Returns:
            Decoded text
        """
        if skip_special:
            special_ids = set(self._special_token_ids.values())
            tokens = [t for t in tokens if t not in special_ids]

        return self.sp.Decode(tokens)

    def encode_chat(
        self,
        messages: List[dict],
        add_generation_prompt: bool = False
    ) -> tuple:
        """
        Encode chat messages to token IDs with target mask.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            add_generation_prompt: Add assistant prompt at end for generation

        Returns:
            Tuple of (token_ids, target_mask) where target_mask is 1 for
            assistant tokens (positions where loss should be computed)
        """
        tokens = []
        target_mask = []

        for msg in messages:
            role = msg['role']
            content = msg['content']

            # Format: <|im_start|>role\ncontent<|im_end|>
            header = f"<|im_start|>{role}\n"
            footer = "<|im_end|>\n"

            header_tokens = self.encode(header)
            content_tokens = self.encode(content)
            footer_tokens = self.encode(footer)

            # Add tokens
            tokens.extend(header_tokens)
            tokens.extend(content_tokens)
            tokens.extend(footer_tokens)

            # Mask: only compute loss on assistant content (not header/footer)
            is_assistant = (role == 'assistant')
            target_mask.extend([0] * len(header_tokens))
            target_mask.extend([1 if is_assistant else 0] * len(content_tokens))
            target_mask.extend([0] * len(footer_tokens))

        # Optionally add generation prompt
        if add_generation_prompt:
            prompt = "<|im_start|>assistant\n"
            prompt_tokens = self.encode(prompt)
            tokens.extend(prompt_tokens)
            target_mask.extend([0] * len(prompt_tokens))

        return tokens, target_mask

    def get_stop_tokens(self) -> List[int]:
        """Get token IDs that should stop generation."""
        stops = []
        if self.im_end_id is not None:
            stops.append(self.im_end_id)
        if self.eos_id is not None:
            stops.append(self.eos_id)
        return stops

    def piece_to_id(self, piece: str) -> int:
        """Get token ID for a piece."""
        return self.sp.PieceToId(piece)

    def id_to_piece(self, id: int) -> str:
        """Get piece for a token ID."""
        return self.sp.IdToPiece(id)


def train_tokenizer(
    input_files: Union[str, List[str]],
    output_prefix: str,
    vocab_size: int = 1024,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    add_chat_tokens: bool = True
) -> str:
    """
    Train a SentencePiece tokenizer.

    Args:
        input_files: Path(s) to training text files
        output_prefix: Output path prefix (will create .model and .vocab)
        vocab_size: Vocabulary size
        model_type: "bpe" or "unigram"
        character_coverage: Character coverage ratio
        add_chat_tokens: Add ChatML special tokens

    Returns:
        Path to trained model file
    """
    if isinstance(input_files, str):
        input_files = [input_files]

    # Build special tokens list (exclude pad/unk/bos/eos as they're built-in)
    user_defined_symbols = []
    if add_chat_tokens:
        builtin = {'pad', 'unk', 'bos', 'eos'}
        user_defined_symbols = [v for k, v in SPECIAL_TOKENS.items() if k not in builtin]

    # Prepare input argument
    input_arg = ','.join(input_files)

    # Train tokenizer
    spm.SentencePieceTrainer.Train(
        input=input_arg,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=SPECIAL_TOKENS['pad'],
        unk_piece=SPECIAL_TOKENS['unk'],
        bos_piece=SPECIAL_TOKENS['bos'],
        eos_piece=SPECIAL_TOKENS['eos'],
        user_defined_symbols=user_defined_symbols,
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=False,
    )

    model_path = f"{output_prefix}.model"
    print(f"Tokenizer trained and saved to: {model_path}")
    return model_path


def test_tokenizer():
    """Test tokenizer functionality (requires trained model)."""
    print("Testing tokenizer...")

    # Create a simple test corpus
    test_corpus = Path("data/test_corpus.txt")
    test_corpus.parent.mkdir(parents=True, exist_ok=True)

    with open(test_corpus, 'w', encoding='utf-8') as f:
        f.write("Hello world! This is a test.\n")
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("Machine learning is fascinating.\n")
        f.write("<|im_start|>user\nWhat is AI?<|im_end|>\n")
        f.write("<|im_start|>assistant\nAI stands for artificial intelligence.<|im_end|>\n")

    # Train a small test tokenizer
    model_path = train_tokenizer(
        input_files=str(test_corpus),
        output_prefix="data/test_tokenizer",
        vocab_size=256,
        add_chat_tokens=True
    )

    # Load and test
    tokenizer = Tokenizer(model_path)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Test basic encode/decode
    text = "Hello, how are you?"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")

    # Test chat encoding
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello!"},
    ]

    tokens, mask = tokenizer.encode_chat(messages)
    print(f"\nChat tokens: {tokens[:20]}...")
    print(f"Target mask: {mask[:20]}...")
    print(f"Assistant token count: {sum(mask)}")

    # Cleanup
    test_corpus.unlink()
    Path("data/test_tokenizer.model").unlink()
    Path("data/test_tokenizer.vocab").unlink()

    print("\nTokenizer tests passed!")


if __name__ == "__main__":
    test_tokenizer()
