from typing import Any, Dict
from graph.chains.retrieval_grader import retrival_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether a document are relevant to the question
    If any document is not relevant, we will set a flag to run web search
    Args:
        state(dict): The current state of the graph
    Returns:
        state(dict): Filtered out irrelevant documents and updated web_search state
    """
    question = state["question"]
    documents = state["documents"]
    filtered_documents = []
    web_search = False
    for d in documents:
        score = retrival_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score

        if grade.lower() == "yes":
            filtered_documents.append(d)
        else:
            web_search = true
            continue
    return {"question": question, "documents": filtered_documents, "web_search": web_search}
