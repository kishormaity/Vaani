"""Benchmark Evaluator for The Relay.

Implements core evaluation metrics:
- Citation Precision, Recall, and F1 (exact match on arXiv ID and page numbers)
- Refusal Behavior Verification
- Grounded Answer Accuracy (factual token recall / overlap against ground truth)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Common English stop words to filter out for factual token overlap scoring
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "of", "while", "during",
    "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "would", "could", "it",
    "its", "by", "with", "at", "into", "onto"
}

# Common refusal language patterns
REFUSAL_TRIGGERS = [
    r"cannot be found in the provided",
    r"not contain sufficient",
    r"does not contain sufficient",
    r"provided passages do not contain",
    r"do not contain (?:any|sufficient|this) information",
    r"does not contain (?:any|sufficient|this) information",
    r"no information provided in the",
    r"unable to (?:answer|find)",
    r"cannot answer",
    r"not mentioned in the",
    r"do not contain sufficient evidence",
    r"out of scope",
]


def _normalize_arxiv_id(raw_id: str) -> str:
    """Normalize arXiv identifier by removing version suffixes and whitespace."""
    clean = re.sub(r"^\[?arXiv:\s*", "", raw_id.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"[,\].]*$", "", clean).strip()
    # Strip version suffix (e.g., 1706.03762v7 -> 1706.03762)
    return re.sub(r"v\d+$", "", clean)


def _extract_citation_pairs(citations: Any) -> Set[Tuple[str, int]]:
    """Extract set of (normalized_paper_id, page_number) pairs from diverse inputs.

    Accepts:
    - Lists of dicts: [{'arxiv_id': '1706.03762v7', 'page': 4}]
    - Lists of strings: ['[arXiv:1706.03762v7, p.4]', '[arXiv:2410.00037, pp.2, 3]']
    - Lists of tuples: [('1706.03762', 4)]
    - Raw text containing citation tags: 'The model uses attention [arXiv:1706.03762v7, p.4].'
    """
    pairs: Set[Tuple[str, int]] = set()

    if not citations:
        return pairs

    # If passed as a single string containing one or more citations
    if isinstance(citations, str):
        # Find all [arXiv:...] tags
        matches = re.findall(r"\[arXiv:[^\]]+\]", citations)
        if not matches:
            matches = [citations]
        items = matches
    elif isinstance(citations, (list, tuple, set)):
        items = citations
    else:
        items = [citations]

    for item in items:
        if isinstance(item, dict):
            raw_id = item.get("arxiv_id") or item.get("paper_id") or ""
            page = item.get("page") or item.get("page_number")
            if raw_id and page is not None:
                norm_id = _normalize_arxiv_id(str(raw_id))
                try:
                    pairs.add((norm_id, int(page)))
                except (ValueError, TypeError):
                    pass

        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            norm_id = _normalize_arxiv_id(str(item[0]))
            try:
                pairs.add((norm_id, int(item[1])))
            except (ValueError, TypeError):
                pass

        elif isinstance(item, str):
            # Parse [arXiv:1706.03762v7, p.5] or [arXiv:2410.00037v1, pp.2, 3]
            match = re.search(r"\[?arXiv:\s*([^,\]]+)(?:,\s*([^\]]+))?\]?", item)
            if match:
                raw_id = match.group(1).strip()
                norm_id = _normalize_arxiv_id(raw_id)
                pages_part = match.group(2)
                if pages_part:
                    cleaned_pages = re.sub(r"pp?\.\s*", "", pages_part.strip())
                    range_match = re.search(r"(\d+)\s*-\s*(\d+)", cleaned_pages)
                    if range_match:
                        start, end = int(range_match.group(1)), int(range_match.group(2))
                        for p in range(start, end + 1):
                            pairs.add((norm_id, p))
                    else:
                        for num_str in re.findall(r"\d+", cleaned_pages):
                            pairs.add((norm_id, int(num_str)))
                else:
                    # No page specified, record with 0 or skip
                    pass

    return pairs


def compute_citation_metrics(
    predicted_citations: Any,
    expected_citations: Any,
) -> Tuple[float, float, float]:
    """Compute exact Citation Precision, Recall, and F1 score on (paper_id, page).

    Args:
        predicted_citations: Predicted citation tags, dicts, or strings.
        expected_citations: Ground truth expected citation tags, dicts, or strings.

    Returns:
        Tuple of (precision, recall, f1) rounded to 4 decimal places.
    """
    pred_set = _extract_citation_pairs(predicted_citations)
    exp_set = _extract_citation_pairs(expected_citations)

    # Edge cases
    if not exp_set and not pred_set:
        # Both empty (expected refusal or unanswerable query with no citations)
        return 1.0, 1.0, 1.0
    if not exp_set and pred_set:
        # Unexpected citations generated
        return 0.0, 1.0, 0.0
    if exp_set and not pred_set:
        # Expected citations missed completely
        return 0.0, 0.0, 0.0

    true_positives = len(pred_set & exp_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(exp_set) if exp_set else 0.0

    if precision + recall > 0.0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_refusal(predicted_refusal: bool, should_refuse: bool) -> bool:
    """Evaluate whether refusal behavior matched expectation.

    Args:
        predicted_refusal: Whether the pipeline refused to answer.
        should_refuse: Whether ground truth requires refusal.

    Returns:
        True if prediction correctly matches expectation, False otherwise.
    """
    return bool(predicted_refusal) == bool(should_refuse)


def _tokenize_for_eval(text: str) -> List[str]:
    """Tokenize text into lowercase content terms, filtering stop words and punctuation."""
    # Remove inline citations [arXiv:...]
    cleaned = re.sub(r"\[arXiv:[^\]]+\]", " ", text)
    # Extract alphanumeric words and numbers
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.\^\/]+\b", cleaned.lower())
    # Filter stop words and short symbols
    filtered = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return filtered


def score_answer_accuracy(predicted_answer: str, ground_truth: str) -> float:
    """Evaluate grounded factual token recall / semantic overlap against reference answer.

    Args:
        predicted_answer: Generated LLM response.
        ground_truth: Gold standard ground truth reference answer.

    Returns:
        Factual overlap score between 0.0 and 1.0.
    """
    if not predicted_answer or not ground_truth:
        return 0.0

    pred_lower = predicted_answer.lower()
    gt_lower = ground_truth.lower()

    # Check if ground truth represents a refusal
    gt_is_refusal = any(re.search(pat, gt_lower) for pat in REFUSAL_TRIGGERS)
    pred_is_refusal = any(re.search(pat, pred_lower) for pat in REFUSAL_TRIGGERS)

    if gt_is_refusal:
        return 1.0 if pred_is_refusal else 0.0
    elif pred_is_refusal:
        return 0.0

    # Tokenize both texts into content terms
    gt_tokens = _tokenize_for_eval(ground_truth)
    pred_tokens = _tokenize_for_eval(predicted_answer)

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.5

    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)

    intersection = gt_set & pred_set
    if not intersection:
        return 0.0

    recall = len(intersection) / len(gt_set)
    precision = len(intersection) / len(pred_set)

    # Compute F1 as harmonic mean of factual token recall and precision
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(f1, 4)
