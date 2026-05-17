"""LFUCacheE: Least-Frequently-Used cache for GraphPrompter embeddings.

Caches high-confidence query embeddings per class label for retrieval-augmented
inference.  Ported from the original GraphPrompter (KDD 2025) reference.
"""

from __future__ import annotations

import torch


class _Node:
    def __init__(self, key: int = 0, val: int = 0, freq: int = 0, embed: torch.Tensor | None = None):
        self.key = key
        self.val = val
        self.embed = embed
        self.freq = freq
        self.prev: _Node | None = None
        self.next: _Node | None = None


class _DoubleList:
    def __init__(self):
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def add_first(self, node: _Node) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node  # type: ignore[union-attr]
        self.head.next = node

    def remove(self, node: _Node) -> None:
        node.prev.next = node.next  # type: ignore[union-attr]
        node.next.prev = node.prev  # type: ignore[union-attr]
        node.next = None
        node.prev = None

    def remove_last(self) -> _Node | None:
        if self.is_empty():
            return None
        last = self.tail.prev
        self.remove(last)
        return last

    def is_empty(self) -> bool:
        return self.head.next == self.tail


class LFUCacheE:
    """LFU cache keyed by embedding sum, storing (embedding, label) pairs.

    Args:
        capacity: Maximum number of cached embeddings.
        ways: Number of classes (used to group cached embeddings by label).
    """

    def __init__(self, capacity: int, ways: int):
        self.cache: dict[int, _Node] = {}
        self.freq: dict[int, _DoubleList] = {}
        self.ncap = capacity
        self.size = 0
        self.min_freq = 0
        self.ways = ways

    def _hash(self, embed: torch.Tensor) -> int:
        return int(round(embed.sum().item(), 8) * 1e8)

    def get(self, embed: torch.Tensor) -> None:
        key = self._hash(embed)
        if key not in self.cache:
            return
        node = self.cache[key]
        self._inc_freq(node)

    def get_all(self) -> dict[int, torch.Tensor]:
        """Return all cached embeddings grouped by label."""
        label2embed = {i: torch.tensor([]) for i in range(self.ways)}
        for key in self.cache:
            node = self.cache[key]
            if label2embed[node.val].numel() == 0:
                label2embed[node.val] = node.embed.unsqueeze(0)
            else:
                label2embed[node.val] = torch.cat((label2embed[node.val], node.embed.unsqueeze(0)), dim=0)
        return label2embed

    def put(self, embed: torch.Tensor, value: int) -> None:
        if self.ncap <= 0:
            return
        key = self._hash(embed)
        if key in self.cache:
            node = self.cache[key]
            self._inc_freq(node)
        else:
            if self.size >= self.ncap:
                node = self.freq[self.min_freq].remove_last()
                if node is not None:
                    del self.cache[node.key]
                    self.size -= 1
            x = _Node(key, value, 1, embed)
            self.cache[key] = x
            if 1 not in self.freq:
                self.freq[1] = _DoubleList()
            self.freq[1].add_first(x)
            self.min_freq = 1
            self.size += 1

    def _inc_freq(self, node: _Node) -> None:
        _freq = node.freq
        self.freq[_freq].remove(node)
        if self.min_freq == _freq and self.freq[_freq].is_empty():
            self.min_freq += 1
            del self.freq[_freq]
        node.freq += 1
        if node.freq not in self.freq:
            self.freq[node.freq] = _DoubleList()
        self.freq[node.freq].add_first(node)
