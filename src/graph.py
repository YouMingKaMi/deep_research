from langgraph.graph import START, END, StateGraph

from src.nodes import (search_node, query_generation_node, normalize_results_node, fetch_content_node, build_evidence_node, plan_search_node, report_generation_node, validate_citations_node)
from src.state import ResearchState


def route_after_query(state: ResearchState) -> str:
    if state.get("query_generation_failed", False):
        return "finish"
    if not state["search_query"]:
        return "finish"
    query = state["search_query"][-1]
    if not query.strip():
        return "finish"
    return "search"

def route_after_evidence(state: ResearchState) -> str:
    turn = state["turn"]
    if turn > 0:
        return "make_query"
    else:
        return "finish"



builder = StateGraph(ResearchState)

builder.add_node("make_query", query_generation_node)
builder.add_node("search", search_node)
builder.add_node("normalize", normalize_results_node)
builder.add_node("fetch", fetch_content_node)
builder.add_node("build_evidence", build_evidence_node)
builder.add_node("plan_subquestion", plan_search_node)
builder.add_node("report", report_generation_node)
builder.add_node("validate_cita", validate_citations_node)

builder.add_edge(START, "plan_subquestion")
builder.add_edge("plan_subquestion", "make_query")
builder.add_conditional_edges(
    "make_query",
    route_after_query,
    {
        "search": "search",
        "finish": END,
    }
                              )
builder.add_edge("search", "normalize")
builder.add_edge("normalize", "fetch")
builder.add_edge("fetch", "build_evidence")
builder.add_conditional_edges(
    "build_evidence",
    route_after_evidence,
    {
        "make_query": "make_query",
        "finish": "report"
    }
)
builder.add_edge("report", "validate_cita")
builder.add_edge("validate_cita", END)


graph = builder.compile()