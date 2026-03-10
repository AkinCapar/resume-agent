from typing import List, TypedDict

class GraphState(TypedDict):
    """
        Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents

    """

    question: str                   # User's initial query
    generation: str                 # Final response produced by the LLM
    web_search: bool                # Determines if it will search on web
    documents: List[str]            # Relevant snippets retrieved from MD files
    search_query: str               # Controlled websearch query
    retries: int