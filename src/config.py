"""Global pipeline config, loaded from config.yaml.

Keeping tunable values (chunk size, overlap, etc.) here instead of hardcoded in the
pipeline modules lets us snapshot different config.yaml versions for evaluation runs
later without touching code.
"""

from dataclasses import dataclass

import yaml


@dataclass
class ChunkingConfig:
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    tokenizer_encoding: str
    module_label_prefix: str


@dataclass
class EmbeddingConfig:
    model: str
    batch_size: int


@dataclass
class VectorStoreConfig:
    collection_name: str
    vector_size: int
    distance: str
    host: str
    port: int


@dataclass
class RetrievalConfig:
    overfetch_k: int
    score_threshold: float
    min_k: int
    max_k: int
    filter_fields: list[str]


@dataclass
class GenerationConfig:
    model: str
    temperature: float
    min_top_score: float


@dataclass
class Config:
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig


def load_config(path: str = "config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        chunking=ChunkingConfig(**raw["chunking"]),
        embedding=EmbeddingConfig(**raw["embedding"]),
        vector_store=VectorStoreConfig(**raw["vector_store"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        generation=GenerationConfig(**raw["generation"]),
    )
