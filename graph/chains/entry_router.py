from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class EntryRouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        description = "Given a user question choose to route it to web search or a vectorstore."
        )
    
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
structured_llm_router = llm.with_structured_output(EntryRouteQuery)

system = """You are an expert at routing a user question to a vectorstore or websearch. 
The vectorstore contains documents related to a person Akin Capar. 
If you get questions with adjectives like "You", "Your", "He" or "His" consider it as they asking about Akin Capar.
Use the vectorstore for questions on Akin Capar. For all else, use web-search."""

entry_route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


question_entry_router = entry_route_prompt | structured_llm_router