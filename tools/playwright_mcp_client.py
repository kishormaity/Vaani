"""Playwright MCP Client communicating with the Playwright MCP Server via stdio.

Enables headless browser interaction, navigation, and web content extraction
using the official Model Context Protocol (MCP) standard over JSON-RPC stdio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check MCP availability
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


@dataclass
class PlaywrightMCPConfig:
    """Configuration for Playwright MCP server execution."""

    command: str = "npx"
    args: List[str] = field(default_factory=lambda: ["-y", "@playwright/mcp@latest"])
    env: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    @classmethod
    def from_config_file(cls, config_path: Optional[Path] = None) -> PlaywrightMCPConfig:
        """Load MCP server parameters from mcp_config.json if available."""
        if config_path is None:
            # Look in project root
            root = Path(__file__).resolve().parent.parent
            config_path = root / "mcp_config.json"

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pw_server = data.get("mcpServers", {}).get("playwright", {})
                cmd = pw_server.get("command", "npx")
                args = pw_server.get("args", ["-y", "@playwright/mcp@latest"])
                env = pw_server.get("env", {})
                return cls(command=cmd, args=args, env=env)
            except Exception as e:
                logger.warning("Failed to parse %s: %s. Using default MCP config.", config_path, e)

        return cls()


class PlaywrightMCPClient:
    """Client for executing Playwright actions through the Playwright MCP Server."""

    def __init__(self, config: Optional[PlaywrightMCPConfig] = None) -> None:
        self.config = config or PlaywrightMCPConfig.from_config_file()

    @property
    def is_supported(self) -> bool:
        """Check if Python mcp library is available."""
        return _HAS_MCP

    def is_available(self) -> bool:
        """Check if both mcp package and node/npx command are present on host."""
        if not self.is_supported:
            return False
        cmd_path = shutil.which(self.config.command)
        return cmd_path is not None

    def _resolve_command(self) -> str:
        """Resolve command binary name to absolute path or executable."""
        cmd = shutil.which(self.config.command)
        if cmd is not None:
            return cmd
        # Windows fallback for npx
        if os.name == "nt" and not self.config.command.lower().endswith((".cmd", ".exe")):
            cmd_with_ext = shutil.which(f"{self.config.command}.cmd")
            if cmd_with_ext:
                return cmd_with_ext
        return self.config.command

    async def fetch_page_text(self, url: str, timeout: Optional[float] = None) -> str:
        """Navigate to a URL using Playwright MCP and extract its text content.

        Args:
            url: Target URL to navigate to.
            timeout: Timeout in seconds for the complete operation.

        Returns:
            Extracted text content from the target web page.
        """
        if not self.is_available():
            raise RuntimeError(
                f"Playwright MCP server is not available (mcp installed: {_HAS_MCP}, "
                f"command '{self.config.command}' found: {shutil.which(self.config.command) is not None})"
            )

        resolved_cmd = self._resolve_command()
        eff_timeout = timeout or self.config.timeout_seconds

        clean_url = url.strip()
        if not clean_url.startswith(("http://", "https://")):
            clean_url = "https://" + clean_url

        server_params = StdioServerParameters(
            command=resolved_cmd,
            args=self.config.args,
            env={**os.environ, **self.config.env},
        )

        logger.info("Connecting to Playwright MCP server via stdio (%s %s)", resolved_cmd, " ".join(self.config.args))

        try:
            async with asyncio.timeout(eff_timeout):
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        logger.info("Playwright MCP session initialized. Navigating to: %s", clean_url)

                        # Step 1: Navigate to target URL
                        nav_result = await session.call_tool(
                            name="browser_navigate",
                            arguments={"url": clean_url},
                        )

                        # Step 2: Extract text using browser_snapshot or browser_evaluate
                        # Try browser_snapshot first for clean accessibility tree text
                        extracted_text = ""
                        try:
                            snap_result = await session.call_tool(
                                name="browser_snapshot",
                                arguments={},
                            )
                            if snap_result and snap_result.content:
                                text_blocks = [
                                    item.text for item in snap_result.content if hasattr(item, "text") and item.text
                                ]
                                extracted_text = "\n".join(text_blocks).strip()
                        except Exception as snap_err:
                            logger.debug("browser_snapshot failed: %s; falling back to evaluate", snap_err)

                        if not extracted_text:
                            # Fallback: evaluate document.body.innerText
                            eval_result = await session.call_tool(
                                name="browser_evaluate",
                                arguments={"function": "() => document.body.innerText || document.title"},
                            )
                            if eval_result and eval_result.content:
                                text_blocks = [
                                    item.text for item in eval_result.content if hasattr(item, "text") and item.text
                                ]
                                extracted_text = "\n".join(text_blocks).strip()

                        # Step 3: Cleanly close page / browser
                        try:
                            await session.call_tool(name="browser_close", arguments={})
                        except Exception:
                            pass

                        return extracted_text

        except TimeoutError as te:
            logger.error("Playwright MCP operation timed out after %.1fs for %s", eff_timeout, clean_url)
            raise TimeoutError(f"Playwright MCP timed out after {eff_timeout}s for {clean_url}") from te
        except Exception as e:
            logger.error("Playwright MCP tool execution error for %s: %s", clean_url, e)
            raise RuntimeError(f"Playwright MCP failure: {e}") from e

