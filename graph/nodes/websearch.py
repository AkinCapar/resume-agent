from typing import Any, Dict
from langchain_classic.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

websearch_tool = TavilySearch(max_results=3)

def websearch(state: GraphState) -> Dict[str,Any]:
    print("---WEB SEARCH---")

    question = state["question"] 
    search_query = state["search_query"] 
    documents = state.get("documents")

    forbidden_words = ["akin", "akın", "capar", "çapar"]
    
    
    if search_query == "NO_SEARCH" or any(word in search_query.lower() for word in forbidden_words):
        print("SECURITY ALERT / NO SEARCH: Skipping Web Search!")
        return {"documents": documents, "web_search": False}

    tavily_results = websearch_tool.invoke({"query": search_query})['results']
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "search_query": search_query, "question": question}