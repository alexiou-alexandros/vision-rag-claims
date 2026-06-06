"""Linear LangGraph pipeline: detect → severity → retrieve → coverage → report."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from vision_rag_claims.agent.nodes import (
    assess_severity_node,
    check_coverage_node,
    detect_damage_node,
    generate_report_node,
    retrieve_policy_node,
)
from vision_rag_claims.agent.state import AgentState


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("detect_damage", detect_damage_node)
    graph.add_node("assess_severity", assess_severity_node)
    graph.add_node("retrieve_policy", retrieve_policy_node)
    graph.add_node("check_coverage", check_coverage_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "detect_damage")
    graph.add_edge("detect_damage", "assess_severity")
    graph.add_edge("assess_severity", "retrieve_policy")
    graph.add_edge("retrieve_policy", "check_coverage")
    graph.add_edge("check_coverage", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
