from typing import Any, Dict
from graph.chains.query_rewriter import query_rewriter
from graph.state import GraphState


def rewrite_query(state: GraphState) -> Dict[str, Any]:
    """
    Analyzes the user's original question and the retrieved documents from the database (RAG).
    If the required information is missing, it generates a clean, general search query 
    suitable for web search by removing personal data like 'Akin'.

    Args:
        state (GraphState): The current state of the graph (contains 'question' and 'documents').

    Returns:
        dict: Returns the newly generated search query ('search_query') to update the state.
    """
    print("---REWRITE QUERY FOR WEB SEARCH---")

    question = state["question"]
    documents = state.get("documents", [])

    result = query_rewriter.invoke({
        "question": question, 
        "documents": documents
    })
    
    print(f"---OPTIMIZED QUERY: {result.query}---")
    
    return {"search_query": result.query}