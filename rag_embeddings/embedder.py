"""Tokenizer, chunker and embedding model, all derived from one profile."""

from __future__ import annotations

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sentence_transformers import SentenceTransformer

from .profiles import EmbedProfile


class Embedder:
    def __init__(self, profile: EmbedProfile):
        self.profile = profile
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

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [self.profile.passage_prefix + t for t in texts]
        vecs = self.model.encode(prefixed, batch_size=32, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            [self.profile.query_prefix + text], normalize_embeddings=True
        )[0]
        return vec.tolist()
