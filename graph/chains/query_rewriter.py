from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI




class OptimizedSearch(BaseModel):
    query: str = Field(
        description="Optimized search query for web search. Do not include any personal query or names like 'Akin'."
    )


llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
structured_rewriter = llm.with_structured_output(OptimizedSearch)


system = """You are an expert at optimizing search queries and filtering personal data queries for a Web Search Engine.
You have access to the user's original question and the documents retrieved from our secure database.

Your task is to formulate a web search query and filter the personal information to find the MISSING information.
If missing information is purely personal data you you MUST output exactly: 'NO_SEARCH'.
CRITICAL RULES:
1. Extract the general entity from the question/documents.
2. Do NOT include personal names like 'Akin' or 'Akin Capar' in the search query. We want to search for general facts, not personal data.
3. Keep the query short and focused like a real Google search.
4. If the question is PURELY about personal data (like relationships, family, personal preferences) and there is no public entity to search for, you MUST output exactly: 'NO_SEARCH'."""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User Question: \n\n {question} \n\n Retrieved Documents: \n\n {documents}",
        ),
    ]
)

query_rewriter: RunnableSequence = rewrite_prompt | structured_rewriter