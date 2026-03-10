from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE, WEBSEARCH, GENERATE, GRADE_DOCUMENTS, REWRITE_QUERY
from graph.nodes.websearch import websearch
from graph.nodes.retrieve import retrieve
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.rewrite_query import rewrite_query
from graph.state import GraphState
from graph.chains.entry_router import question_entry_router, EntryRouteQuery
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader

load_dotenv()

def decide_to_generate(state):
    print ("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: ALL DOCUMENTS ARE IRRELEVANT TO QUESTION---"
        )

        return WEBSEARCH
    
    else:
        print("---DECISION: GENERATE---")
        return GENERATE
    
def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: EntryRouteQuery = question_entry_router.invoke({"question": question})

    if source.datasource == WEBSEARCH: 
        print("---ROUTE QUESTION TO WEB SEARCH---")  
        return WEBSEARCH
    
    elif source.datasource == "vectorstore": 
        print("---ROUTE QUESTION TO RAG---")  
        return RETRIEVE
    
def grade_generation_gounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    retries = state.get("retries", 0)

    if retries >= 3:
        print("---MAX RETRIES REACHED: BREAKING INFINITE LOOP---")
        return "max_retries"
        

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, websearch)
workflow.add_node(REWRITE_QUERY, rewrite_query)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: REWRITE_QUERY,
        RETRIEVE: RETRIEVE
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: REWRITE_QUERY,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_gounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": REWRITE_QUERY,
        "max_retries": END
    }
)

workflow.add_edge(REWRITE_QUERY, WEBSEARCH)
workflow.add_edge(WEBSEARCH, GENERATE)
#workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")