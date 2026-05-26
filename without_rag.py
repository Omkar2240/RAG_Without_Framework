import pypdf
import chromadb
from chromadb.utils import embedding_functions
import os
import uuid
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# ------------------------ Ingestion Pipline ----------------------------------

def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def split_text(text, chunk_size=500):
    sentences = text.replace("\n", " ").split(". ")  # it removes \n and split (new line) when . (full stop + space) occurs
    
#     Input:
#     Python is easy. AI is powerful. NLP is interesting.

#     After split:
#     [
#      "Python is easy",
#      "AI is powerful",
#      "NLP is interesting."
#     ]

    chunks = []
    current_chunk = []
    current_size = 0
    
    
    for sentence in sentences:
        sentence = sentence.strip() #removes leading and trailing spaces
        
        # skips empty sentence
        if not sentence:
            continue
        
        # if '.' not, then add it 
        if not sentence.endswith('.'):
            sentence += '.'
            
        sentence_size = len(sentence)
        
        if sentence_size + chunk_size > chunk_size and current_size:    #  used "and current_size" because, if current_size is empty, then it will return false, so will excute else block.  it prevents adding empty sentence in chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks


db_client = chromadb.PersistentClient(path="chroma-db-no-rag")
    
sentence_transformer = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
      )
    
    # Create or get existing collection
collection = db_client.get_or_create_collection(
        name="documents_collection",
        embedding_function=sentence_transformer
    )


def create_vector_store(chunks, file_name):
    print("creating embeddings and storing in vector db")
    
    ids = []
    documents = []
    metadata = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())  # creates unique id for each chunk
        
        ids.append(chunk_id)
        documents.append(chunk)
        
        metadata.append({
            "source": file_name,
            "chunk_number" : i
        })
        
    # store inside chroma
    collection.add(
        ids = ids,
        documents = documents,
        metadatas = metadata
    )
    
    print(f"Saved {len(chunks)} chunks in the chroma db")          


# ------------------------------ Retrival and Generation ------------------------------------            

def semantic_search(query, n_results):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results

def get_context_with_sources(results):
    
    # Combine document chunks into a single context
    context = "\n\n".join(results['documents'][0])

    # Format sources with metadata
    sources = [
        f"{meta['source']} (chunk {meta['chunk_number']})" 
        for meta in results['metadatas'][0]
    ]

    return context, sources



API_KEY = os.getenv("HF_TOKEN")

client = InferenceClient(api_key=API_KEY)

def get_prompt(context, query):
    
    prompt = f"""Based on following context.Please provide the relevent and contextual response.If the answer cannot 
    be derived from the context then say 
    "I cannot answer this based on the provided information." 
    
    Context from Document: {context}
    
    Human: {query}
    
    Assistant:
    
    """
    
    return prompt

def generate_response(query, context):
    prompt = get_prompt(context, query)
    
    try:
        responses = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are helpful assistant, who answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0, # more focused
            max_tokens=500
        )
        
        return responses.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def rag_query(query,  n_chunks= 2):
    results = semantic_search(query, n_chunks)
    context, sources = get_context_with_sources(results)
    
    response = generate_response(query, context)
    
    return response, sources

def main():
    pdf_path = "docs/Microsoft.pdf"
    file_name = "Microsoft.pdf"
    pdf_text = read_pdf(pdf_path)
    # print(pdf_text[:500])  # Print the first 500 characters of the PDF text
    
    # chunks = split_text(pdf_text)
    
    # create_vector_store(chunks, file_name)
    
    query = "Who is CEO of Microsoft?"
    
    response, sources = rag_query(query)

    # Print results
    print("\nQuery:", query)
    print("\nAnswer:", response)
    print("\nSources used:")
    for source in sources:
        print(f"- {source}")
    
    
if __name__ == "__main__":
    main()