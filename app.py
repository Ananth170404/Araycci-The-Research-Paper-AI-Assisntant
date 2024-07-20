import streamlit as st
from streamlit_player import st_player
from rohit_section import generate_response_from_chunks, get_relevant_chunks, create_index, extract_text_from_pdf, clean_text, chunk_text, store_chunks_in_pinecone
from translate import translate, generate_audio

# Display the custom logo using st.image
st.sidebar.image("logo.jpg")
st.title("Aryacci Research Paper Bot")
st.sidebar.title("PDF Research Assistant")

if 'video_played' not in st.session_state:
    st.session_state.video_played = False

if not st.session_state.video_played:
    start_button = st.button("Start")
    
    if start_button:
        st_player("https://drive.google.com/file/d/1_YOCEvRtVfex7Zkh4G4PGnQpcO2Qqw_z/view?pli=1")  # Replace with your video URL
        st.session_state.video_played = True
else:
    pdf_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    lang = st.sidebar.radio("Choose", ["English", "French", "Spanish"])

    if pdf_files:
        if 'index' not in st.session_state:
            st.session_state.index = None

        if st.session_state.index is None:
            with st.spinner("Processing PDFs..."):
                combined_chunks = []
                for pdf_file in pdf_files:
                    text = extract_text_from_pdf(pdf_file)
                    cleaned_text = clean_text(text)
                    chunks = chunk_text(cleaned_text)
                    combined_chunks.extend(chunks)
                
                st.session_state.index = create_index()
                if st.session_state.index:
                    store_chunks_in_pinecone(combined_chunks, st.session_state.index)
                    st.success("PDFs processed and indexed successfully!")
                else:
                    st.error("Failed to create Pinecone index.")
        
        if st.session_state.index:
            query = st.text_input("Enter your question:", key="query")
            ask_button = st.button("Ask")
            end_button = st.button("End conversation")

            if ask_button and query:
                with st.spinner("Searching for answers..."):
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

            if end_button:
                st.session_state.index = None
                st.session_state.query = ""
                st.experimental_rerun()

    else:
        st.info("Please upload one or more PDF files to start.")
