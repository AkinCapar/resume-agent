# 📄 AI Resume & Portfolio Agent (Advanced RAG)

An intelligent Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **LangChain**, and **OpenAI**. This agent acts as a digital assistant that answers questions based strictly on an uploaded PDF (Resume/CV) document. 

## 🌟 Key Features

* **🧠 Advanced RAG Architecture:** Doesn't just blindly pass text to an LLM. It retrieves, grades, and validates the documents before generating an answer.
* **🚦 Controlled Workflow (State Machine):** Uses LangGraph to implement a custom state machine with conditional routing. The flow is strictly controlled to ensure high-quality outputs.
* **🔍 Document Grading & Filtering:** The system evaluates retrieved document chunks for relevance. If a chunk is irrelevant to the user's query, it is discarded to prevent the LLM from getting confused.
* **🌐 Fallback Web Search:** If the required information is not found in the provided PDF, the agent intelligently routes the query to a web search node to find the answer externally.
* **🔒 Zero Hallucination Policy:** Driven by a strict System Prompt, the agent is constrained to answer *only* based on the provided facts, making it highly reliable for professional use cases.

## 🏗️ Architecture (LangGraph Flow)

The system is orchestrated using a custom **StateGraph** with distinct nodes and conditional edges:
1.  **`retrieve_node`:** Extracts relevant context from the vector database using similarity search.
2.  **`grade_documents_node`:** An LLM-based grader that evaluates if the retrieved text actually answers the prompt.
3.  **`web_search_node`:** A conditional fallback that uses a Tavily search API if the local documents lack the necessary information.
4.  **`generate_node`:** The final step where the LLM synthesizes the validated context into a concise, accurate response.

## 🛠️ Tech Stack

* **Framework:** LangGraph / LangChain
* **LLM:** OpenAI 
* **Vector Store:** ChromaDB 
* **Web search tool:** Tavily 
* **Embeddings:** OpenAI Embeddings
* **UI:** Streamlit