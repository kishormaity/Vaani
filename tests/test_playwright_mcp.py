"""Unit tests for Playwright MCP Client integration and routing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.classifier import ClassificationResult, QueryIntent
from pipeline.orchestrator import RAGOrchestrator
from tools.playwright_mcp_client import PlaywrightMCPConfig, PlaywrightMCPClient
from tools.web_fetcher import fetch_web_page


class TestPlaywrightMCPClient:
    """Test Playwright MCP client configuration, initialization, and fallbacks."""

    def test_config_loading_from_file(self, tmp_path: Path):
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(
            json.dumps({
                "mcpServers": {
                    "playwright": {
                        "command": "npx",
                        "args": ["-y", "@playwright/mcp@latest"]
                    }
                }
            }),
            encoding="utf-8"
        )
        cfg = PlaywrightMCPConfig.from_config_file(config_file)
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "@playwright/mcp@latest"]

    def test_client_is_supported(self):
        client = PlaywrightMCPClient()
        assert client.is_supported is True
        # On a system with node/npx, is_available should be boolean
        assert isinstance(client.is_available(), bool)

    @pytest.mark.anyio
    async def test_fetch_web_page_mcp_mock_success(self):
        """Verify fetch_web_page returns cleaned text when MCP succeeds."""
        mock_text = "<html><body><h1>Title</h1><p>Test content from MCP</p></body></html>"
        with patch.object(PlaywrightMCPClient, "is_available", return_value=True), \
             patch.object(PlaywrightMCPClient, "fetch_page_text", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_text

            result = await fetch_web_page("https://example.com", prefer_mcp=True)
            assert "Test content from MCP" in result
            mock_fetch.assert_awaited_once()

    @pytest.mark.anyio
    async def test_fetch_web_page_mcp_fallback_on_error(self):
        """Verify fetch_web_page falls back to HTTP or mock when MCP raises an exception."""
        with patch.object(PlaywrightMCPClient, "is_available", return_value=True), \
             patch.object(PlaywrightMCPClient, "fetch_page_text", side_effect=RuntimeError("MCP failed")):

            # Should fall back cleanly without raising
            result = await fetch_web_page("https://example.com/unavailable", timeout=1.0)
            assert isinstance(result, str)
            assert len(result) > 0


class TestPlaywrightMCPOrchestratorRouting:
    """Test orchestrator routing integration for EXTERNAL_WEB with Playwright MCP."""

    @pytest.mark.anyio
    async def test_orchestrator_external_web_routes_and_telemetry(self):
        from llm.standin_service import StandinConfig, StandinLLMService
        from gateway.llm_gateway import LLMGateway

        standin = StandinLLMService(config=StandinConfig(simulate_latency=False, failure_rate=0.0))
        gateway = LLMGateway(client=standin)

        # Mock classifier to force EXTERNAL_WEB
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = ClassificationResult(
            query="Fetch the latest AI news from https://example.org",
            is_in_scope=True,
            question_type="external_web",
            intent=QueryIntent.EXTERNAL_WEB,
            external_url="https://example.org",
        )

        orchestrator = RAGOrchestrator(
            llm_client=standin,
            gateway=gateway,
            classifier=mock_classifier,
        )

        with patch("tools.web_fetcher.fetch_web_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = "Example web context content about AI models."
            result = await orchestrator.run_pipeline("Fetch the latest AI news from https://example.org")

            assert "external_web_fetch" in result.steps_executed
            assert "external_web_synthesize" in result.steps_executed
            assert result.telemetry.get("mcp_server") == "playwright"
            assert result.telemetry.get("web_url") == "https://example.org"
            assert len(result.answer) > 0

