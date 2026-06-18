"""Shared tool layer.

Every topology gets the SAME tools, for the same reason the model is shared:
to isolate the effect of topology. Add SEC EDGAR and a code sandbox in later
phases (categories: quantitative/financial reasoning, tool-use-heavy).
"""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use for recent events, facts,
    companies, and anything not in the model's training data. Returns the top
    results as text."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for r in results
    )


@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for academic papers. Use for research questions, methods,
    and technical topics. Returns titles, authors, and abstracts."""
    import arxiv

    search = arxiv.Search(
        query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance
    )
    out = []
    for r in arxiv.Client().results(search):
        authors = ", ".join(a.name for a in r.authors)
        out.append(f"{r.title}\n{authors}\n{r.summary}")
    return "\n\n".join(out) if out else "No papers found."


# The shared tool set. Topologies import this, never their own.
TOOLS = [web_search, arxiv_search]
