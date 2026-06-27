from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mock-mcp-server")


@mcp.tool(structured_output=True)
def lookup(key: str) -> dict[str, Any]:
    """Look up a value in the mock task database."""
    data = {
        "color": "blue",
        "shape": "circle",
        "owner": "agent-env-smoke",
    }
    if key not in data:
        return {"observation": f"{key} is unknown", "found": False, "done": False, "score": 0.0}
    return {"observation": data[key], "value": data[key], "found": True, "done": False, "score": 0.0}


@mcp.tool(structured_output=True)
def echo(text: str) -> dict[str, Any]:
    """Return the supplied text."""
    return {"observation": text, "done": False, "score": 0.0}


@mcp.tool(structured_output=True)
def finish(answer: str) -> dict[str, Any]:
    """Finish the current mock task with an answer."""
    success = answer.strip().lower() == "blue"
    return {
        "observation": "finished",
        "answer": answer,
        "done": True,
        "success": success,
        "score": 1.0 if success else 0.0,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
