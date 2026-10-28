"""Bounded-memory, reproducible token sampling and assistant-only chat batches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_instruction_jsonl(path):
    """Read physical JSONL lines; Unicode separators inside strings are data."""
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PackedCorpus:
    """Sample complete fixed-size windows, without a corpus-size permutation.

    Sampling is with replacement, weighted by available windows in each source.
    The seed/step/rank/microstep tuple reconstructs batches after a restart.
    Documents are separated by EOS; attention is not reset at each boundary.
    """

    def __init__(self, root, split, seq_len, seed=42):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.seq_len, self.seed = seq_len, seed
        self.arrays = []
        self.names = []
        for source in self.manifest['sources']:
            item = source['files'][split]
            path = self.root / item['path']
            if path.stat().st_size != item['tokens'] * 2:
                raise ValueError(f'Incomplete token file: {path}')
            self.arrays.append(np.memmap(path, mode='r', dtype='<u2'))
            self.names.append(source['name'])
        self.sizes = np.array([(len(array) - 1) // seq_len for array in self.arrays])
        if np.any(self.sizes < 1):
            raise ValueError('Each source needs at least one complete sequence')
        self.boundaries = np.cumsum(self.sizes)

    def batch(self, batch_size, step, rank=0, microstep=0, source=None):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, rank, microstep]))
        if source is None:
            indices = rng.integers(0, self.boundaries[-1], size=batch_size)
            source_ids = np.searchsorted(self.boundaries, indices, side='right')
            local = indices - np.concatenate(([0], self.boundaries[:-1]))[source_ids]
        else:
            source_ids = np.full(batch_size, source)
            local = rng.integers(0, self.sizes[source], size=batch_size)
        windows = np.stack([
            self.arrays[s][i * self.seq_len:i * self.seq_len + self.seq_len + 1]
            for s, i in zip(source_ids, local)
        ]).astype(np.int64)
        return torch.from_numpy(windows[:, :-1].copy()), torch.from_numpy(windows[:, 1:].copy())


def encode_instruction(tokenizer, messages, max_seq_len):
    """Keep complete conversations and supervise assistant text AND turn ending.

    Tokenize each turn as one string, so an inference prompt and training prompt
    have the same boundary. Reject overlong examples instead of cutting answers.
    """
    ids, mask = [], []
    for message in messages:
        role, content = message['role'], message['content']
        if role not in ('system', 'user', 'assistant') or not isinstance(content, str):
            return None
        header = tokenizer.encode(f'<|im_start|>{role}\n')
        turn = tokenizer.encode(f'<|im_start|>{role}\n{content}<|im_end|>\n')
        if turn[:len(header)] != header:
            raise ValueError('Tokenizer does not preserve the chat header boundary')
        ids.extend(turn)
        mask.extend([0] * len(header) + [int(role == 'assistant')] * (len(turn) - len(header)))
    if len(ids) > max_seq_len or not any(mask[1:]):
        return None
    return ids, mask


def instruction_batch(examples, tokenizer, max_seq_len):
    encoded = [encode_instruction(tokenizer, item['messages'], max_seq_len) for item in examples]
    if any(item is None for item in encoded):
        raise ValueError('Invalid or overlong instruction example')
    length = max(len(item[0]) for item in encoded)
    ids = torch.full((len(encoded), length), max(0, tokenizer.pad_id), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.float32)
    for index, (tokens, target_mask) in enumerate(encoded):
        ids[index, :len(tokens)] = torch.tensor(tokens)
        mask[index, :len(tokens)] = torch.tensor(target_mask)
    return ids, ids.clone(), mask
