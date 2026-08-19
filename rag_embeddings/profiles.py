"""Embedding profiles: the one object tokenizer, chunker and embedder derive from."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EmbedProfile:
    model_id: str
    max_tokens: int          # from the model card, never model_max_length
    headroom: int = 128      # contextualize() is applied after the split
    passage_prefix: str = ""
    query_prefix: str = ""

    @property
    def version(self) -> str:
        return f"{self.model_id}@{self.max_tokens}-{self.headroom}"


BGE_M3 = EmbedProfile(model_id="BAAI/bge-m3", max_tokens=8192)

# E5-family models need the prefixes — omitting them degrades retrieval
# quietly rather than loudly.
E5_LARGE = EmbedProfile(
    model_id="intfloat/multilingual-e5-large",
    max_tokens=512,
    passage_prefix="passage: ",
    query_prefix="query: ",
)

PROFILES: dict[str, EmbedProfile] = {
    "bge-m3": BGE_M3,
    "e5-large": E5_LARGE,
}


def resolve(
    name: str,
    *,
    max_tokens: int | None = None,
    headroom: int | None = None,
) -> EmbedProfile:
    """Look a profile up by name, or by HuggingFace model id with `max_tokens`."""
    profile = PROFILES.get(name)
    if profile is None:
        if max_tokens is None:
            raise ValueError(
                f"unknown profile {name!r}; either pick one of "
                f"{sorted(PROFILES)} or pass max_tokens from the model card"
            )
        profile = EmbedProfile(model_id=name, max_tokens=max_tokens)

    overrides = {}
    if max_tokens is not None:
        overrides["max_tokens"] = max_tokens
    if headroom is not None:
        overrides["headroom"] = headroom
    return replace(profile, **overrides) if overrides else profile
