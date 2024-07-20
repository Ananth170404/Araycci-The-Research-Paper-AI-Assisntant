import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient

# Initialize Pinecone
pinecone_api_key = "b887f4da-c8c8-4e25-954b-1c0c15df7312"
pinecone_environment = "us-east-1"
pc = Pinecone(api_key=pinecone_api_key)

# Model initialization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone index name
index_name = "llama3"

def create_index():
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
    pc.create_index(
        name=index_name, 
        dimension=384,
        metric='cosine', 
        spec=ServerlessSpec(
            cloud='aws',
            region=pinecone_environment
        )
    )
    return pc.Index(index_name)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def chunk_text(text, max_chunk_size=512, overlap=128):
    words = text.split(' ')
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + max_chunk_size])
        chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks

def store_chunks_in_pinecone(chunks, index):
    chunk_embeddings = model.encode(chunks)
    vectors = [{"id": f"chunk-{i}", "values": embedding.tolist(), "metadata": {"content": chunk, "type": "chunk"}}
               for i, (embedding, chunk) in enumerate(zip(chunk_embeddings, chunks))]
    index.upsert(vectors)

def get_relevant_chunks(query, index, top_k=5):
    query_embedding = model.encode([query])[0].tolist()
    search_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    chunks = [result['metadata']['content'] for result in search_results['matches']]
    return chunks

def generate_response_from_chunks(chunks, query):
    combined_content = "\n".join([f"Chunk:\n{chunk}" for chunk in chunks])
    prompt_template = (
        "You are an AI research assistant. Your job is to help users understand and extract key insights from research papers. "
        "You will be given a query and context from multiple research papers. Based on this information, provide accurate, concise, and helpful responses. "
        "Here is the context from the research papers and the user's query:\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}\n\n"
        "Please provide a detailed and informative response based on the given context."
    )
    user_query = prompt_template.format(context=combined_content, query=query)
    client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token="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")
    response = client.chat_completion(messages=[{"role": "user", "content": user_query}], max_tokens=500, stream=False)
    return response['choices'][0]['message']['content'] if response['choices'] else "No response received."

def process_pdfs(pdf_files, query):
    index = create_index()
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        store_chunks_in_pinecone(chunks, index)
    relevant_chunks = get_relevant_chunks(query, index)
    response = generate_response_from_chunks(relevant_chunks, query)
    return response
