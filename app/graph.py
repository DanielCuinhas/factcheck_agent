from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import State

BRANCHES = ["origin", "reputation", "authors", "suspicion", "corroboration"]


def build_graph():
    g = StateGraph(State)
    g.add_node("ingest", nodes.ingest)
    g.add_node("extract", nodes.extract)
    for name in BRANCHES:
        g.add_node(name, getattr(nodes, name))
    g.add_node("synthesize", nodes.synthesize)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "extract")
    for name in BRANCHES:               # fan out: the five checks run in parallel
        g.add_edge("extract", name)
    g.add_edge(BRANCHES, "synthesize")  # fan in: wait for all five
    g.add_edge("synthesize", END)
    return g.compile()


graph = build_graph()
