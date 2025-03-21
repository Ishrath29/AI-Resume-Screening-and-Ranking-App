import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit page config
st.set_page_config(page_title="AI Resume Screening", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .big-font { font-size: 22px !important; }
    .small-font { font-size: 14px; color: grey; }
    </style>
    """, unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarities = cosine_similarity([job_desc_vector], resume_vectors).flatten()
    return similarities

# Streamlit UI
st.title("📄 AI Resume Screening & Candidate Ranking System")
st.markdown("<p class='small-font'>Upload resumes in PDF format and rank candidates based on job description similarity.</p>", unsafe_allow_html=True)

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔹 Job Description")
    job_description = st.text_area("Enter the job description:", height=150)

with col2:
    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.subheader("📊 Ranking Resumes")
    resumes = []
    progress_bar = st.progress(0)

    for idx, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        resumes.append(text)
        progress_bar.progress((idx + 1) / len(uploaded_files))

    scores = rank_resumes(job_description, resumes)
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score (%)": scores * 100})
    results = results.sort_values(by="Score (%)", ascending=False)

    # Apply color formatting to the score column
    def highlight_score(val):
        """Function to color code scores."""
        if val > 70:
            color = 'green'  # High similarity
        elif val > 40:
            color = 'orange '  # Medium similarity
        else:
            color = 'red'  # Low similarity
        return f'background-color: {color}; color: black;'

    # Display styled dataframe
    st.dataframe(results.style.format({"Score (%)": "{:.2f}"}).map(highlight_score, subset=["Score (%)"]))
