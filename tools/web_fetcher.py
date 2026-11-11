"""Web Fetcher tool providing Playwright MCP Server integration with graceful HTTP fallback."""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Check direct Playwright library availability
try:
    from playwright.async_api import async_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False


def _clean_html_text(raw_html: str, max_chars: int = 4000) -> str:
    """Strip HTML tags, scripts, styles, and collapse whitespace."""
    if not raw_html:
        return ""
    # Remove script and style blocks
    cleaned = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", " ", raw_html, flags=re.IGNORECASE)
    cleaned = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", " ", cleaned, flags=re.IGNORECASE)
    # Remove all HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Decode basic HTML entities
    cleaned = (
        cleaned.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


async def fetch_web_page(
    url: str,
    timeout: float = 15.0,
    prefer_mcp: bool = True,
) -> str:
    """Fetch clean plain text content from an external web page.

    1. Attempts Playwright MCP server execution over stdio (@playwright/mcp).
    2. Falls back to direct async_playwright headless browser if installed.
    3. Falls back to asynchronous HTTP request via httpx.
    4. Falls back to deterministic mock content if the URL is unreachable.

    Args:
        url: URL of the web page to fetch.
        timeout: Maximum seconds to wait.
        prefer_mcp: Whether to attempt Playwright MCP server first (default: True).

    Returns:
        Clean plain text content extracted from the web page.
    """
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    # 1. Attempt Playwright MCP Server
    if prefer_mcp:
        try:
            from tools.playwright_mcp_client import PlaywrightMCPClient

            mcp_client = PlaywrightMCPClient()
            if mcp_client.is_available():
                logger.info("Attempting web page fetch via Playwright MCP server for %s", clean_url)
                text = await mcp_client.fetch_page_text(clean_url, timeout=timeout)
                if text and text.strip():
                    return _clean_html_text(text)
        except Exception as e:
            logger.warning(
                "Playwright MCP fetch failed for %s (%s). Falling back to direct browser/HTTP client.",
                clean_url,
                e,
            )

    # 2. Attempt direct Playwright headless browser
    if _HAS_PLAYWRIGHT:
        try:
            logger.info("Attempting web page fetch via direct Playwright for %s", clean_url)
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    page = await browser.new_page()
                    await page.goto(clean_url, timeout=int(timeout * 1000))
                    body_text = await page.inner_text("body")
                    if body_text and body_text.strip():
                        return _clean_html_text(body_text)
                finally:
                    await browser.close()
        except Exception as e:
            logger.warning(
                "Direct Playwright fetch failed for %s (%s). Falling back to HTTP client.",
                clean_url,
                e,
            )

    # 3. Attempt HTTP client fallback via httpx
    try:
        import httpx

        logger.info("Attempting web page fetch via httpx for %s", clean_url)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            headers = {
                "User-Agent": "TheRelayResearchAssistant/1.0 (Mozilla/5.0 compatible)"
            }
            response = await client.get(clean_url, headers=headers)
            response.raise_for_status()
            return _clean_html_text(response.text)
    except Exception as e:
        logger.warning(
            "httpx fetch failed for %s (%s). Returning fallback mock content.",
            clean_url,
            e,
        )

    # 4. Graceful fallback text if external network/page is unreachable
    return f"[Web Content from {clean_url}: Unable to retrieve live page ({clean_url}). Standard fallback content.]"
