"""smart_memory: a governed, four-store memory layer for AI agents.

Quick start::

    from smart_memory import Memory

    mem = Memory("agent.db")
    alice = mem.scope(user_id="alice", agent_id="support")
    alice.add("Alice prefers email over phone calls")
    print(alice.context("How should I contact Alice?"))
"""

from .context import ContextBuilder, ContextCard
from .embeddings import Embedder, FunctionEmbedder, HashingEmbedder, SentenceTransformerEmbedder
from .memory import EmbedderMismatchError, Memory, ScopedMemory
from .ontology import Glossary, KnowledgeGraph, Relation, Term
from .refine import Refiner, RefineReport
from .retrieval import HybridRetriever, RetrievalConfig
from .router import CallableRouter, Router, RuleRouter
from .storage import InMemoryStore, PermissionDenied, SQLiteStore, StorageBackend
from .traces import TraceLog
from .types import MemoryRecord, MemoryType, RoutingDecision, Scope, SearchResult, Status
from .working import WorkingMemory

__version__ = "0.1.0"

__all__ = [
    "Memory", "ScopedMemory", "EmbedderMismatchError",
    "MemoryType", "MemoryRecord", "Scope", "SearchResult", "RoutingDecision", "Status",
    "Embedder", "HashingEmbedder", "FunctionEmbedder", "SentenceTransformerEmbedder",
    "Router", "RuleRouter", "CallableRouter",
    "StorageBackend", "SQLiteStore", "InMemoryStore", "PermissionDenied",
    "HybridRetriever", "RetrievalConfig", "ContextBuilder", "ContextCard",
    "KnowledgeGraph", "Glossary", "Relation", "Term",
    "Refiner", "RefineReport", "TraceLog", "WorkingMemory",
    "__version__",
]
