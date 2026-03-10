from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever 
from graph.chains.entry_router import question_entry_router, EntryRouteQuery
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.query_rewriter import query_rewriter



#def test_retrieval_grader_answer_yes() -> None:
#    question = "agent memory"
#    docs = retriever.invoke(question)
#    doc_txt = docs[1].page_content
#
#    res: GradeDocuments = retrieval_grader.invoke(
#        {"question": question, "document": doc_txt}
#    )
#
#    assert res.binary_score == "yes"
#
#def test_retrieval_grader_answer_no() -> None:
#    question = "agent memory"
#    docs = retriever.invoke(question)
#    doc_txt = docs[1].page_content
#
#    res: GradeDocuments = retrieval_grader.invoke(
#        {"question": "how to make a pizza?", "document": doc_txt}
#    )
#
#    assert res.binary_score == "no"
#
#def test_router_to_vectorstore() -> None:
#    question = "what are his hobbies?"
#
#    res: EntryRouteQuery = question_entry_router.invoke({"question": question})
#
#    assert res.datasource == "vectorstore"
#
#def test_router_to_websearch() -> None:
#    question = "how to play chess?"
#
#    res: EntryRouteQuery = question_entry_router.invoke({"question": question})
#
#    assert res.datasource == "websearch"
#
#def test_generation_chain() -> None:
#    question = "What are the colleges you graduated from? And give me some information about what year these colleges established?"
#    docs = retriever.invoke(question)
#
#    generation = generation_chain.invoke({"context": docs, "question": question})
#    pprint(generation)
#
def test_hallucination_grader_answer_yes() -> None:
    question = "How old is akin?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert res.binary_score

#def test_hallucination_grader_answer_no() -> None:
#    question = "What degrees are Akin holding?"
#    docs = retriever.invoke(question)
#    generation = generation_chain.invoke({"context": docs, "question": question})
#
#    res: GradeHallucinations = hallucination_grader.invoke(
#        {
#            "documents": docs,
#            "generation": "Berlin is the capital of Germany.",
#        }
#    )
#
#    assert not res.binary_score

#def test_query_rewriter() -> None:
#    question = "what is Akin's girlfriends name?"
#    docs = retriever.invoke(question)
#
#    query = query_rewriter.invoke({"question": question, "documents": docs})
#
#    pprint(query)