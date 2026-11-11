"""Query Classifier for The Relay RAG pipeline.

Identifies question intent/type, routes queries across:
1. CORPUS_RAG: Queries concerning the 3 indexed papers (Attention, Moshi, Dynamic Chunking).
2. EXTERNAL_WEB: Queries requesting live web data, URL navigation, or browser-assisted fetching.
3. DIRECT_LLM: General everyday queries, chit-chat, programming help, or common knowledge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Routing destinations for user queries."""

    CORPUS_RAG = "CORPUS_RAG"
    EXTERNAL_WEB = "EXTERNAL_WEB"
    DIRECT_LLM = "DIRECT_LLM"


# Canonical papers and associated terminology
VERIFIED_PAPERS: Dict[str, Dict[str, Any]] = {
    "1706.03762": {
        "title": "Attention Is All You Need",
        "keywords": {
            "attention", "transformer", "self-attention", "multi-head", "multihead",
            "scaled dot-product", "dot product", "encoder", "decoder", "positional encoding",
            "positional encodings", "feed-forward", "wmt", "vaswani", "bleu",
            "layer normalization", "residual connection", "warmup", "sub-layer",
            "query", "key", "value"
        },
    },
    "2410.00037": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "keywords": {
            "moshi", "mimi", "helium", "speech", "dialogue", "full-duplex", "audio",
            "spoken", "text-to-speech", "speech-to-text", "conversational", "latency",
            "acoustic", "codec", "kyutai", "defossez", "duplex", "voice", "streaming audio"
        },
    },
    "2507.07955": {
        "title": "Dynamic Chunking for Retrieval-Augmented Generation",
        "keywords": {
            "dynamic chunking", "chunking", "rag", "retrieval-augmented", "chunk", "chunks",
            "segmentation", "boundary", "boundaries", "token window", "semantic chunking",
            "chunk size", "overlap", "hierarchical chunking", "hierarchical sequence modeling",
            "transition entropy"
        },
    },
}

# Explicitly out-of-scope topics / domains for corpus RAG
OUT_OF_SCOPE_TRIGGERS: Set[str] = {
    "quantum", "relativity", "photosynthesis", "sourdough", "recipe", "pizza",
    "cooking", "cookies", "cookie", "bake", "baking", "cake", "capital of",
    "weather in", "stock market", "cryptocurrency", "bitcoin", "football",
    "cricket", "nba", "soccer", "president of", "movie", "cancer", "surgery",
    "dna", "chemistry", "reactjs", "kubernetes", "css", "gardening",
    "horoscope", "astrology"
}

# External web triggers
WEB_FETCH_PATTERNS = [
    r"\b(fetch|browse|navigate|scrape|visit)\s+(https?://|www\.|web|url|site|website|page)\b",
    r"\b(search\s+the\s+web|web\s+search|look\s*up\s+on\s+the\s+web|search\s+online)\b",
    r"\b(on\s+the\s+web|from\s+the\s+web|browse\s+to|check\s+the\s+website)\b",
    r"\b(playwright|fetch\s+page|fetch\s+web\s+page)\b",
]

# Chit-chat and greeting patterns
CHITCHAT_PATTERNS = [
    r"^(hello|hi|hey|greetings|howdy|good\s+(morning|afternoon|evening|day))[\s!.,?]*$",
    r"\b(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|introduce\s+yourself)\b",
    r"\b(how\s+are\s+you|how's\s+it\s+going|what's\s+up)\b",
    r"\b(thank\s+you|thanks|bye|goodbye|see\s+ya)\b",
    r"\b(tell\s+me\s+a\s+joke|make\s+me\s+laugh)\b",
]

# General programming assistance patterns
PROGRAMMING_PATTERNS = [
    r"\b(write|create|implement|debug|code|generate)\s+(a\s+)?(python|javascript|typescript|c\+\+|java|rust|go|bash|sql|html|css)?\s*(function|script|code|algorithm|program|class|method|query)\b",
    r"\b(how\s+to|how\s+do\s+i)\s+(sort|reverse|filter|parse|read|write|implement|calculate)\b",
    r"\b(fibonacci|palindrome|quicksort|merge\s*sort|binary\s*search|linked\s*list|reverse\s+a\s+string)\b",
    r"\b(python\s+code|python\s+script|bash\s+script|sql\s+query|syntax\s+error|debug\s+this)\b",
]


@dataclass
class ClassificationResult:
    """Outcome of query classification and routing."""

    query: str
    is_in_scope: bool
    question_type: str  # "factual", "comparison", "explanation", "chit_chat", "programming", "external_web"
    intent: QueryIntent = QueryIntent.CORPUS_RAG
    target_papers: List[str] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""
    external_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "is_in_scope": self.is_in_scope,
            "question_type": self.question_type,
            "intent": self.intent.value if hasattr(self.intent, "value") else str(self.intent),
            "target_papers": self.target_papers,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
            "external_url": self.external_url,
        }


class QueryClassifier:
    """Fast rule/keyword-assisted query classifier and route selector."""

    def __init__(self, fallback_llm: Optional[Any] = None) -> None:
        self.fallback_llm = fallback_llm

    def classify(self, query: str) -> ClassificationResult:
        """Classify query intent into CORPUS_RAG, EXTERNAL_WEB, or DIRECT_LLM.

        Args:
            query: Input user query.

        Returns:
            ClassificationResult containing intent route, question type, target papers, and scope flag.
        """
        clean_query = query.strip()
        lower_query = clean_query.lower()

        # Step 1: Check for External Web Routing (URLs or web navigation requests)
        url_match = re.search(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", clean_query, re.IGNORECASE)
        has_web_phrase = any(re.search(pat, lower_query) for pat in WEB_FETCH_PATTERNS)

        if url_match or has_web_phrase:
            external_url = None
            if url_match:
                extracted = url_match.group(0)
                if extracted.lower().startswith("www."):
                    extracted = "https://" + extracted
                external_url = extracted

            return ClassificationResult(
                query=clean_query,
                is_in_scope=True,
                question_type="external_web",
                intent=QueryIntent.EXTERNAL_WEB,
                target_papers=[],
                confidence=0.95,
                reasoning="Query requests external web fetching or live site navigation.",
                external_url=external_url,
            )

        # Step 2: Detect Target Papers in Corpus
        matched_papers: List[str] = []
        for paper_id, meta in VERIFIED_PAPERS.items():
            if paper_id in lower_query:
                matched_papers.append(paper_id)
                continue

            keywords: Set[str] = meta["keywords"]
            for kw in keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, lower_query):
                    matched_papers.append(paper_id)
                    break

        matched_papers = sorted(list(set(matched_papers)))

        # Step 3: Check Out-of-Scope Triggers (domains explicitly non-corpus)
        matched_out_of_scope: List[str] = []
        for trigger in OUT_OF_SCOPE_TRIGGERS:
            pattern = r"\b" + re.escape(trigger) + r"\b"
            if re.search(pattern, lower_query):
                matched_out_of_scope.append(trigger)

        # Step 4: Check Chit-Chat and Programming Triggers
        is_chitchat = any(re.search(pat, lower_query) for pat in CHITCHAT_PATTERNS)
        is_programming = any(re.search(pat, lower_query) for pat in PROGRAMMING_PATTERNS)

        # Step 5: Route Decision
        if matched_papers:
            # Matches verified corpus paper(s) -> Route: CORPUS_RAG
            is_in_scope = True
            question_type = self._detect_question_type(lower_query)
            reasoning = f"Query matched verified paper(s): {', '.join(matched_papers)}."
            intent = QueryIntent.CORPUS_RAG
            confidence = 0.95

        elif matched_out_of_scope:
            # Clearly out-of-scope domain -> Route: CORPUS_RAG refusal
            is_in_scope = False
            question_type = self._detect_question_type(lower_query)
            reasoning = f"Query mentions out-of-scope domain(s): {', '.join(matched_out_of_scope)}."
            intent = QueryIntent.CORPUS_RAG
            confidence = 0.90

        elif is_chitchat:
            # Chit-chat greeting or conversational query -> Route: DIRECT_LLM
            is_in_scope = True
            question_type = "chit_chat"
            reasoning = "Query classified as conversational chit-chat."
            intent = QueryIntent.DIRECT_LLM
            confidence = 0.95

        elif is_programming:
            # General programming / coding assistance -> Route: DIRECT_LLM
            is_in_scope = True
            question_type = "programming"
            reasoning = "Query classified as general programming assistance."
            intent = QueryIntent.DIRECT_LLM
            confidence = 0.90

        else:
            # Check for general academic/ML terminology
            academic_terms = {"paper", "author", "metric", "loss", "accuracy", "benchmark", "model", "evaluation"}
            has_academic_term = any(re.search(r"\b" + re.escape(term) + r"\b", lower_query) for term in academic_terms)
            if has_academic_term:
                is_in_scope = True
                question_type = self._detect_question_type(lower_query)
                reasoning = "Query contains general ML/research terminology relevant to the corpus."
                intent = QueryIntent.CORPUS_RAG
                confidence = 0.70
            else:
                # Default to DIRECT_LLM for general everyday queries not mentioning corpus
                is_in_scope = True
                question_type = self._detect_question_type(lower_query)
                reasoning = "Query classified as general common knowledge / everyday inquiry."
                intent = QueryIntent.DIRECT_LLM
                confidence = 0.75

        return ClassificationResult(
            query=clean_query,
            is_in_scope=is_in_scope,
            question_type=question_type,
            intent=intent,
            target_papers=matched_papers,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _detect_question_type(self, text: str) -> str:
        """Determine question type from syntax and keywords."""
        # Comparison triggers
        comp_patterns = [
            r"\bcompare\b", r"\bdifference between\b", r"\bvs\.?\b", r"\bversus\b",
            r"\bdiffer from\b", r"\btradeoff\b", r"\btrade-off\b", r"\bbetter than\b",
            r"\badvantages of\b"
        ]
        if any(re.search(p, text) for p in comp_patterns):
            return "comparison"

        # Explanation triggers
        expl_patterns = [
            r"\bwhy\b", r"\bhow does\b", r"\bhow do\b", r"\bexplain\b",
            r"\bdescribe\b", r"\bmechanism\b", r"\bworkflow\b", r"\bprocess\b"
        ]
        if any(re.search(p, text) for p in expl_patterns):
            return "explanation"

        # Factual triggers
        fact_patterns = [
            r"\bwhat is\b", r"\bwhat are\b", r"\bwhich\b", r"\bwho\b",
            r"\bwhen\b", r"\bdefine\b", r"\bhow many\b", r"\bformula\b",
            r"\bequation\b", r"\bdimension\b", r"\bparameter\b"
        ]
        if any(re.search(p, text) for p in fact_patterns):
            return "factual"

        return "factual"
