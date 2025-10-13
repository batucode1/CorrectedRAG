from graph.chains.generation import generation_chain
from typing import Dict, Any


def generate(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke(
        {"context": "documents", "question": "question"}
    )
    return
