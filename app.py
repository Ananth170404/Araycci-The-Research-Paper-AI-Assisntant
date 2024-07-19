import streamlit as st
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from gtts import gTTS
import io

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
    # Open the PDF file-like object using PyMuPDF
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
        "You will be given a query and context from the research paper. Based on this information, provide accurate, concise, and helpful responses. "
        "Here is the context from the research paper and the user's query:\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}\n\n"
        "Please provide a detailed and informative response based on the given context."
    )
    user_query = prompt_template.format(context=combined_content, query=query)
    client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token="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")
    response = client.chat_completion(messages=[{"role": "user", "content": user_query}], max_tokens=500, stream=False)
    return response['choices'][0]['message']['content'] if response['choices'] else "No response received."

def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io

# Streamlit app
st.sidebar.title("PDF Research Assistant")
pdf_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    if 'index' not in st.session_state:
        st.session_state.index = None

    if st.session_state.index is None:
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(pdf_file)
            cleaned_text = clean_text(text)
            chunks = chunk_text(cleaned_text)
            
            # Create Pinecone index and store chunks
            st.session_state.index = create_index()
            if st.session_state.index:
                store_chunks_in_pinecone(chunks, st.session_state.index)
                st.success("PDF processed and indexed successfully!")
            else:
                st.error("Failed to create Pinecone index.")
    
    # Query handling
    if st.session_state.index:
        query = st.text_input("Enter your question:")
        if st.button("Ask"):
            with st.spinner("Searching for answers..."):
                relevant_chunks = get_relevant_chunks(query, st.session_state.index)
                response = generate_response_from_chunks(relevant_chunks, query)
                
                # Display text response
                st.write(response)
                
                # Generate and display audio response
                audio_io = generate_audio(response)
                st.audio(audio_io, format='audio/mp3')
                st.download_button(label="Download Audio Response", data=audio_io, file_name="response.mp3", mime="audio/mp3")
        
        if st.button("Ask another question"):
            st.experimental_rerun()
        
        if st.button("End conversation"):
            st.session_state.index = None
            st.stop()  # Close the Streamlit app
