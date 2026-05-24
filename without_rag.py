import pypdf
import chromadb
from chromadb.utils import embedding_functions
import os
import uuid

# load document and extract text
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



client = chromadb.PersistentClient(path="chroma-db-no-rag")
    
sentence_transformer = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
      )
    
    # Create or get existing collection
collection = client.get_or_create_collection(
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
            
        
        
    


def main():
    pdf_path = "docs/Microsoft.pdf"
    file_name = "Microsoft.pdf"
    pdf_text = read_pdf(pdf_path)
    print(pdf_text[:500])  # Print the first 500 characters of the PDF text
    
    chunks = split_text(pdf_text)
    
    print(create_vector_store(chunks, file_name))
    
    
    
if __name__ == "__main__":
    main()