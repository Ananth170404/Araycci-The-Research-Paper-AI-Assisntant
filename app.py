import streamlit as st
from RAG import generate_response_from_chunks, get_relevant_chunks, create_index, extract_text_from_pdf, clean_text, chunk_text, store_chunks_in_pinecone
from translate import translate, generate_audio
from ARAYCCI_ALL import search_arxiv, process_docs, clustering, sanitize_filename, list_pdfs
import os

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'search' not in st.session_state:
    st.session_state.search = []
if 'query' not in st.session_state:
    st.session_state.query = None
if 'download' not in st.session_state:
    st.session_state.download = False
if 'papers_downloaded' not in st.session_state:
    st.session_state.papers_downloaded = False
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'fig' not in st.session_state:
    st.session_state.fig = None

def reset_page():
    st.session_state.index = None
    st.session_state.search = []
    st.session_state.query = None
    st.session_state.papers_downloaded = False
    st.session_state.result_df = None
    st.session_state.fig = None

# Streamlit app
st.sidebar.image("logo.jpg")
st.title("Aryacci Research Paper Assistant")
st.sidebar.title("Research Assistant")

lang = st.sidebar.radio("Choose", ["English", "French", "Spanish"])

Source = st.radio(
    "Pick Source of Papers",
    ["Local", "Web"],
    index=0,
    on_change=reset_page
)

# Language map
language_map = {
    'English': 'en-US',
    'Spanish': 'es-ES',
    'French': 'fr-FR'
}

def process_local_pdfs(data):
    combined_chunks = []
    for pdf_file in data:
        text = extract_text_from_pdf(pdf_file)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        combined_chunks.extend(chunks)
    return combined_chunks

def download_and_process_arxiv(selection, arxiv_results, save_dir):
    if not os.path.exists(save_dir) and save_dir:
        os.makedirs(save_dir)
    result_df, processed_documents = process_docs(selection, arxiv_results, save_dir)
    result_df, fig = clustering(result_df, processed_documents)
    st.session_state.result_df = result_df
    st.session_state.fig = fig
    return result_df, fig

def handle_query_response(query, lang):
    relevant_chunks = get_relevant_chunks(query, st.session_state.index)
    response = generate_response_from_chunks(relevant_chunks, query)
    if lang != "English":
        translated_response = translate(response, lang)
        st.write(translated_response)
        audio_io = generate_audio(translated_response, lang)
    else:
        st.write(response)
        audio_io = generate_audio(response, lang)
    st.audio(audio_io, format='audio/mp3')
    st.download_button(label="Download Audio Response", data=audio_io, file_name="response.mp3", mime="audio/mp3")

if Source == "Local":
    data = st.sidebar.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)
    if data:
        if not st.session_state.papers_downloaded:
            with st.spinner("Processing PDFs..."):
                combined_chunks = process_local_pdfs(data)
                st.session_state.index = create_index()
                if st.session_state.index:
                    store_chunks_in_pinecone(combined_chunks, st.session_state.index)
                    st.session_state.papers_downloaded = True
                    st.success("PDF processed and indexed successfully!")
                else:
                    st.error("Failed to create Pinecone index.")
elif Source == "Web":
    search = st.text_input("Enter the search query: ")
    max_results = st.slider("Maximum results:", 10, 100)
    if st.button("Search"):
        st.session_state.search = search_arxiv(search, max_results)
    if st.session_state.search:
        arxiv_results = st.session_state.search
        selection = {}
        for i, result in enumerate(arxiv_results):
            st.subheader(f"{i+1}. {result['title']} ({result['published']})")
            st.write(f"**Authors:** {', '.join(result['authors'])}")
            st.write(f"**Summary:** {result['summary']}")
            st.write(f"**Link:** [arXiv Paper]({result['link']})")
            selection[i] = st.checkbox("Download Paper", key=str(i+1))
        selected_indices = [i for i in selection if selection[i]]
        save_dir = st.text_input("Enter the directory to save PDFs and images: ")
        if st.button("Download Selection") and save_dir:
            st.session_state.download = True
        if st.session_state.download:
            if not st.session_state.papers_downloaded:
                with st.spinner("Downloading and processing papers..."):
                    result_df, fig = download_and_process_arxiv(selected_indices, arxiv_results, save_dir)
                    st.session_state.papers_downloaded = True
                st.pyplot(st.session_state.fig)
                st.write(st.session_state.result_df)
            selected_cluster = st.text_input("Enter Cluster number")
            if selected_cluster:
                if st.session_state.result_df is not None:
                    cluster_docs = st.session_state.result_df[st.session_state.result_df['Cluster'] == int(selected_cluster)]
                    save_cluster_dir = os.path.join(save_dir, f"Cluster_{selected_cluster}")
                    if not os.path.exists(save_cluster_dir):
                        os.makedirs(save_cluster_dir)
                    for index, row in cluster_docs.iterrows():
                        pdf_path = row['PDF File']
                        sanitized_pdf_filename = sanitize_filename(os.path.basename(pdf_path))
                        with open(os.path.join(save_cluster_dir, f"{sanitized_pdf_filename}.pdf"), 'wb') as pdf_file:
                            pdf_file.write(open(pdf_path, 'rb').read())
                    data = list_pdfs(save_cluster_dir)
                    if not st.session_state.papers_downloaded:
                        with st.spinner("Processing PDFs..."):
                            combined_chunks = process_local_pdfs(data)
                            st.session_state.index = create_index()
                            if st.session_state.index:
                                store_chunks_in_pinecone(combined_chunks, st.session_state.index)
                                st.session_state.papers_downloaded = True
                                st.success("PDF processed and indexed successfully!")
                            else:
                                st.error("Failed to create Pinecone index.")

# Query handling
if st.session_state.index:
    query = st.text_input("Enter your question:")
    if query:
        st.session_state.query = query
    if st.button("Ask") and st.session_state.query:
        with st.spinner("Searching for answers..."):
            handle_query_response(st.session_state.query, lang)
        st.session_state.query = ""
    if st.button("End conversation"):
        reset_page()
        st.experimental_rerun()
