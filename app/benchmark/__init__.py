"""Benchmark harness: registry loading, run logging, and optional Ollama judge."""

from app.benchmark.registry import BenchmarkRegistry, load_registry

__all__ = ["BenchmarkRegistry", "load_registry"]
