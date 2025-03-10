import streamlit as st
import os
from werkzeug.utils import secure_filename
from pdf_to_text_extraction import extract_text_from_pdf, setup_gemini_api
from summarization import summarize_text
from rag import run_rag_query #importing the run_rag_query from rag.py
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PDF_PATH = Path("data/The_Gift_of_the_Magi.pdf")
DEFAULT_OUTPUT_PATH = Path("output/extracted_text.txt")
DEFAULT_SUMMARY_PATH = Path("output/summary.txt")

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_pdf(pdf_path, output_path, summary_path, api_key):
    """Processes a PDF, extracts text, summarizes, and returns paths."""
    setup_gemini_api(api_key)
    extracted_text = extract_text_from_pdf(pdf_path, api_key)
    if not extracted_text:
        st.error("Error extracting text from PDF.")
        return None, None

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    summary = summarize_text(extracted_text, api_key)
    if not summary:
        st.error("Error summarizing text.")
        return None, None

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    return output_path, summary_path

st.title("PDF Summarizer & Question Answering")

api_key = st.text_input("Enter your Google Gemini API Key:", type="password")

if not api_key:
    st.warning("Please enter your API key to proceed.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key #setting the api key as an environment variable.

if "summary" not in st.session_state:
    st.session_state.summary = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = None
if "output_path" not in st.session_state:
    st.session_state.output_path = None

pdf_source = st.radio("Select PDF Source:", ("The Gift of the Magi (Default)", "Upload Your Own PDF"))

uploaded_file = None
if pdf_source == "Upload Your Own PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if st.button("Process PDF"):
    if pdf_source == "The Gift of the Magi (Default)":
        st.session_state.output_path, summary_path = process_pdf(DEFAULT_PDF_PATH, DEFAULT_OUTPUT_PATH, DEFAULT_SUMMARY_PATH, api_key)
        if st.session_state.output_path and summary_path:
            with open(DEFAULT_SUMMARY_PATH, 'r', encoding='utf-8') as f:
                st.session_state.summary = f.read()
            st.session_state.pdf_filename = "The_Gift_of_the_Magi.pdf"

    elif uploaded_file is not None:
        filename = secure_filename(uploaded_file.name)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        upload_output_path = Path(OUTPUT_FOLDER) / f"{filename.split('.')[0]}_extracted.txt"
        upload_summary_path = Path(OUTPUT_FOLDER) / f"{filename.split('.')[0]}_summary.txt"

        st.session_state.output_path, summary_path = process_pdf(Path(filepath), upload_output_path, upload_summary_path, api_key)

        if st.session_state.output_path and summary_path:
            with open(upload_summary_path, 'r', encoding='utf-8') as f:
                st.session_state.summary = f.read()
            st.session_state.pdf_filename = filename

    if st.session_state.summary:
        st.subheader("Summary:")
        st.write(st.session_state.summary)

        with st.form("question_form"):
            question = st.text_input("Ask a Question:")
            submitted = st.form_submit_button("Ask")

            if submitted:
                if "conversation_history" not in st.session_state:
                    st.session_state.conversation_history = []

                answer = run_rag_query(question, st.session_state.output_path) #using run_rag_query from rag.py

                if answer:
                    st.session_state.conversation_history.append({"question": question, "answer": answer})
                    st.subheader("Answer:")
                    st.write(answer)
                else:
                    st.error("Failed to get an answer. Please check the console for errors.")

                st.subheader("Conversation History")
                for item in st.session_state.conversation_history:
                    st.write(f"Question: {item['question']}")
                    st.write(f"Answer: {item['answer']}")