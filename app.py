import streamlit as st
from RAG import generate_response_from_chunks, get_relevant_chunks, create_index, extract_text_from_pdf, clean_text, store_chunks_in_pinecone, combined_chunking
from translate import translate, generate_audio
from stt import start_recording, stop_recording
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
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'save_dir' not in st.session_state:
    st.session_state.save_dir = None
if 'selected_indices' not in st.session_state:
    st.session_state.selected_indices = []

def reset_page():
    st.session_state.index = None
    st.session_state.search = []
    st.session_state.query = None
    st.session_state.papers_downloaded = False
    st.session_state.result_df = None
    st.session_state.fig = None
    st.session_state.selected_cluster = None
    st.session_state.save_dir = None
    st.session_state.selected_indices = []

# Streamlit app
st.sidebar.image("logo.jpg")
st.title("Arayacci Research Paper Bot")
st.sidebar.title("PDF Research Assistant")

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
        chunks = combined_chunking(cleaned_text)
        combined_chunks.extend(chunks)
    return combined_chunks

def download_and_process_arxiv(selection, arxiv_results, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_df, processed_documents = process_docs(selection, arxiv_results, save_dir)
    result_df, fig = clustering(result_df, processed_documents)
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

# Handle Local PDF Processing
if Source == "Local":
    data = st.sidebar.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)
    if data:
        if not st.session_state.papers_downloaded:
            st.write("Processing PDFs...")
            with st.spinner("Processing PDFs..."):
                combined_chunks = process_local_pdfs(data)
                st.session_state.index = create_index()
                if st.session_state.index:
                    store_chunks_in_pinecone(combined_chunks, st.session_state.index)
                    st.session_state.papers_downloaded = True
                    st.success("PDF processed and indexed successfully!")
                else:
                    st.error("Failed to create Pinecone index.")

# Handle Web Search and Download
if Source == "Web":
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
        if st.button("Download Selection"):
            if save_dir:
                st.session_state.download = True
                st.session_state.save_dir = save_dir
                st.session_state.selected_indices = selected_indices
                st.write(f"Directory set to: {st.session_state.save_dir}")
                st.write(f"Selected indices: {st.session_state.selected_indices}")
            else:
                st.error("Please enter a directory to save PDFs and images.")
        if st.session_state.download and st.session_state.save_dir and st.session_state.selected_indices:
            st.write("Starting download and processing...")
            if not st.session_state.papers_downloaded:
                with st.spinner("Downloading and processing papers..."):
                    result_df, fig = download_and_process_arxiv(st.session_state.selected_indices, arxiv_results, st.session_state.save_dir)
                    st.session_state.result_df = result_df
                    st.session_state.fig = fig
                    st.session_state.papers_downloaded = True
                    st.session_state.download = False  # Reset the download flag
                    st.success("Papers downloaded and processed successfully!")
                    st.pyplot(st.session_state.fig)
                    st.write(st.session_state.result_df)

        if st.session_state.papers_downloaded:
            selected_cluster = st.text_input("Enter Cluster number")
            if st.button("Process Cluster") and selected_cluster:
                st.write(f"Processing cluster: {selected_cluster}")
                selected_cluster = int(selected_cluster)
                if st.session_state.result_df is not None:
                    cluster_docs = st.session_state.result_df[st.session_state.result_df['Cluster'] == selected_cluster]
                    save_cluster_dir = os.path.join(st.session_state.save_dir, f"Cluster_{selected_cluster}")
                    if not os.path.exists(save_cluster_dir):
                        os.makedirs(save_cluster_dir)
                    for index, row in cluster_docs.iterrows():
                        pdf_path = row['PDF File']
                        sanitized_pdf_filename = sanitize_filename(os.path.basename(pdf_path))
                        with open(os.path.join(save_cluster_dir, f"{sanitized_pdf_filename}.pdf"), 'wb') as pdf_file:
                            pdf_file.write(open(pdf_path, 'rb').read())
                    data = list_pdfs(save_cluster_dir)
                    st.write(f"Processing PDFs in cluster directory: {save_cluster_dir}")
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
    speech = st.checkbox("Voice")
    query = st.text_input("Enter your question:") if not speech else ""
    if speech:
        if lang in language_map:
            language = language_map[lang]
        if st.button("Start Recording"):
            start_recording()
            st.write("Recording...")
        if st.button("Stop Recording"):
            query = stop_recording()
            st.write(query)
    if query:
        st.session_state.query = query
    if st.button("Ask") and st.session_state.query:
        with st.spinner("Searching for answers..."):
            handle_query_response(st.session_state.query, lang)
        st.session_state.query = ""
    if st.button("End conversation"):
        reset_page()
        st.experimental_rerun()
