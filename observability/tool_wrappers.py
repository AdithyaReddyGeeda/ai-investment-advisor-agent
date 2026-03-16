import time
from typing import Any, Dict, List

from langchain_core.tools import BaseTool, tool as tool_factory

from .logging_utils import log_tool_call


def wrap_tools_with_logging(tools: List[BaseTool]) -> List[BaseTool]:
    """Return new tool objects that log each invocation to JSONL."""

    wrapped: List[BaseTool] = []

    for t in tools:
        name = getattr(t, "name", t.__class__.__name__)

        @tool_factory
        def logged_tool(**kwargs: Any) -> Any:  # type: ignore[no-redef]
            start = time.time()
            try:
                result = t.invoke(kwargs)
                duration_ms = (time.time() - start) * 1000.0
                log_tool_call(
                    name=name,
                    args=kwargs,
                    duration_ms=duration_ms,
                    success=True,
                )
                return result
            except Exception as e:  # pragma: no cover - defensive logging
                duration_ms = (time.time() - start) * 1000.0
                log_tool_call(
                    name=name,
                    args=kwargs,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
                raise

        logged_tool.name = name  # type: ignore[attr-defined]
        logged_tool.description = getattr(t, "description", "")  # type: ignore[attr-defined]
        wrapped.append(logged_tool)  # type: ignore[arg-type]

    return wrapped

