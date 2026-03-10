from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

persist_dir = "./.chroma"

def ingest_all_data():
    data_dir = "resume-data"
    
    md_files = [f for f in os.listdir(data_dir) if f.endswith(".md")]

    print(f"---Found md files are: {md_files}---")
    
    if not md_files:
        print(f"---Error: could nor found .md files in '{data_dir}' folder!---")
        return

    all_sections = []

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0)

    for file_name in md_files:
        file_path = os.path.join(data_dir, file_name)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        header_splits = markdown_splitter.split_text(content)
        
        for split in header_splits:
            split.metadata["source_file"] = file_name
            
        final_splits = text_splitter.split_documents(header_splits)
        all_sections.extend(final_splits)
        
        print(f"----Processed: {file_name} -> {len(final_splits)} chunks.----")

        
    with open("debug_chunks.txt", "w", encoding="utf-8") as f:
        f.write(f"TOTAL CHUNK {len(all_sections)}\n")
        f.write("="*50 + "\n\n")
        
        for i, chunk in enumerate(all_sections):
            f.write(f"--- CHUNK {i+1} ---\n")
            f.write(f"METADATA: {chunk.metadata}\n")
            f.write(f"CONTENT:\n{chunk.page_content}\n")
            f.write("-" * 30 + "\n\n")

    print(f"---Debug logs written to debug_chunks.txt---")

    print(f"----\nWriting {len(all_sections)} chunks to ChromaDB...----")
    
    vectorstore = Chroma.from_documents(
        documents=all_sections,
        collection_name="resume-rag",
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir,
    )
    
    print("---All documents have been successfully vectorized and saved!---")

retriever = Chroma(
    collection_name="resume-rag",
    embedding_function=OpenAIEmbeddings(),
    persist_directory=persist_dir,
).as_retriever()

if __name__ == "__main__":
    ingest_all_data()

