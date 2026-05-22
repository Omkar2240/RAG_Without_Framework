import os
from dotenv import load_dotenv
from langchain_core.documents import Document
import pypdf
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def load_pdf_pages(directory: str) -> list[Document]:
    
    if not os.path.isdir(directory):
      print(f"Directory {directory} does not exist.")
      return []

    all_docs = []
    for file_name in os.listdir(directory):
        if file_name.lower().endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            try:
                reader = pypdf.PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    all_docs.append(Document(
                        page_content=page.extract_text() or "",
                        metadata={"source": file_path, "page": i},
                    ))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return all_docs
  
    
    
def split_into_chunks(docs, chunk_size=800, chunk_overlap=0):
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    
    if chunks:
        
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i+1}:")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Chunk content preview: {chunk.page_content[:100]}")
            print("-" * 40)
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma"):
    print("creating embeddings and storing in vector db")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print("---creating vector store---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("---finished creating vector store--- ")
    
    print(f"Vector store created and saved in {persist_directory}")
    
    return vectorstore
    
    

def main():
    # load pdf documents
    docs_dir = "docs"
    
    persistent_directory = "db/chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")
        
        embedding_model = HuggingFaceEmbeddings(
          model_name="sentence-transformers/all-mpnet-base-v2",
          encode_kwargs={"normalize_embeddings": True},
       )
        vectorstore = Chroma.from_documents(
          documents=chunks,
          embedding=embedding_model,
          persist_directory=persistent_directory,
          collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore
    
    print("Persistent directory does not exist. Initializing vector store...\n")
    
    # step 1: load pdf documents
    docs = load_pdf_pages(docs_dir)

    
    # step 2: chunk documents
    chunks = split_into_chunks(docs)
    
    # step 3: create embeddings and store in vector db
    vectorstore = create_vector_store(chunks)
    
    
    print("Main function initialized.")

if __name__ == "__main__":
    main()