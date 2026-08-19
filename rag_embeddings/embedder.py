"""Tokenizer, chunker and embedding model, all derived from one profile."""

from __future__ import annotations

import logging

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBED_TOKEN_BUDGET
from .profiles import EmbedProfile

log = logging.getLogger(__name__)

# Why a token budget and not a batch size: a transformer's peak allocation
# scales with (sequences x padded length), so a fixed count of sequences bounds
# nothing. bge-m3 accepts 8192 tokens, and a batch of 32 of them asks for a
# single feed-forward tensor of 32 x 8192 x 4096 x 4 bytes = 17 GB. That is the
# OOM, and no amount of container memory fixes it — only capping the product
# does. At the default budget the same tensor is 256 MB.
#
# For the ordinary ~512-token chunk this still packs 32 into a batch, which is
# exactly what the fixed batch size gave. It only bites on long chunks, which
# is the case that used to be fatal.


class Embedder:
    def __init__(
        self,
        profile: EmbedProfile,
        token_budget: int = DEFAULT_EMBED_TOKEN_BUDGET,
    ):
        self.profile = profile
        self.token_budget = token_budget
        # max_tokens must be passed: left unset, HuggingFaceTokenizer derives it
        # from the model's config, which is the number the profile exists to
        # override.
        self.tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=profile.model_id,
            max_tokens=profile.max_tokens - profile.headroom,
        )
        self.chunker = HybridChunker(tokenizer=self.tokenizer, merge_peers=True)
        self.model = SentenceTransformer(profile.model_id)

    def count(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def _batches(self, counts: list[int]) -> list[list[int]]:
        """Group indices, longest first, into batches that fit the token budget."""
        order = sorted(range(len(counts)), key=counts.__getitem__, reverse=True)

        batches: list[list[int]] = []
        current: list[int] = []
        width = 0
        for i in order:
            if not current:
                current, width = [i], counts[i]
                continue
            # width is the batch's longest, which every member is padded out to.
            if (len(current) + 1) * width <= self.token_budget:
                current.append(i)
            else:
                batches.append(current)
                current, width = [i], counts[i]
        if current:
            batches.append(current)
        return batches

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [self.profile.passage_prefix + t for t in texts]
        counts = [self.count(t) for t in prefixed]

        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        batches = self._batches(counts)
        log.debug(
            "encoding %d passages in %d batches, budget %d tokens",
            len(texts), len(batches), self.token_budget,
        )
        for batch in batches:
            if len(batch) == 1 and counts[batch[0]] > self.token_budget:
                log.warning(
                    "passage of %s tokens exceeds the %s token budget on its "
                    "own; encoding it alone",
                    counts[batch[0]], self.token_budget,
                )
            vecs = self.model.encode(
                [prefixed[i] for i in batch],
                batch_size=len(batch),
                normalize_embeddings=True,
            )
            for i, vec in zip(batch, vecs):
                out[i] = vec.tolist()
        return out

    def encode_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            [self.profile.query_prefix + text], normalize_embeddings=True
        )[0]
        return vec.tolist()
