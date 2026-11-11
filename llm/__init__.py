"""LLM generation and client interface for The Relay."""

from llm.base import CompletionResult, LLMClientProtocol, RateLimitError, RateLimitTelemetry
from llm.groq_client import GroqLLMClient
from llm.standin_service import StandinConfig, StandinLLMService

__all__ = [
    "CompletionResult",
    "GroqLLMClient",
    "LLMClientProtocol",
    "RateLimitError",
    "RateLimitTelemetry",
    "StandinConfig",
    "StandinLLMService",
]
