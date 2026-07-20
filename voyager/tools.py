"""Shared tools for every topology: web_search and arxiv_search.

web_search prefers Tavily when TAVILY_API_KEY is set — Tavily is a real search
API that works from any IP, including datacenters like Hugging Face Spaces — and
falls back to DuckDuckGo (ddgs) when no key is present. So it runs locally with
no key, and runs *reliably* on the Space once the key is set.
"""
from __future__ import annotations

import os

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Returns the top results as text.

    Args:
        query: The search terms as a plain string, e.g.
            "latest Llama model benchmark results". Required.
    """
    # Prefer Tavily when a key is available (reliable from datacenter IPs).
    if os.getenv("TAVILY_API_KEY"):
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            results = client.search(query, max_results=5).get("results", [])
            if results:
                return "\n\n".join(
                    f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}"
                    for r in results
                )
        except Exception:
            pass  # fall through to DuckDuckGo on any Tavily error

    # Fallback: DuckDuckGo (no key, but rate-limited from shared/datacenter IPs).
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
        if not hits:
            return "No results found."
        return "\n\n".join(
            f"{h.get('title', '')}\n{h.get('href', '')}\n{h.get('body', '')}"
            for h in hits
        )
    except Exception as e:
        return f"web_search error: {e}"


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers. Returns titles, authors, and summaries.

    Args:
        query: The topic or keywords as a plain string, e.g.
            "agent topology evaluation LLM". Required.
    """
    try:
        import arxiv
        search = arxiv.Search(
            query=query, max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        out = []
        for r in arxiv.Client().results(search):
            authors = ", ".join(a.name for a in r.authors[:4])
            out.append(f"{r.title}\n{authors}\n{r.entry_id}\n{r.summary[:400]}")
        return "\n\n".join(out) if out else "No papers found."
    except Exception as e:
        return f"arxiv_search error: {e}"


TOOLS = [web_search, arxiv_search]
