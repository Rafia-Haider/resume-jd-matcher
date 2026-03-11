import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import PyPDF2

st.set_page_config(
    page_title="Smart Resume-JD Matcher",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'r', 'scala',
    'golang', 'ruby', 'php', 'swift', 'kotlin', 'matlab', 'bash', 'perl',
    'machine learning', 'deep learning', 'nlp', 'natural language processing',
    'computer vision', 'reinforcement learning', 'transfer learning',
    'supervised learning', 'unsupervised learning', 'neural network',
    'random forest', 'xgboost', 'gradient boosting', 'svm', 'regression',
    'classification', 'clustering', 'pca', 'feature engineering',
    'feature selection', 'model evaluation', 'hyperparameter tuning',
    'time series', 'anomaly detection', 'recommendation system',
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
    'scipy', 'matplotlib', 'seaborn', 'plotly', 'opencv', 'nltk', 'spacy',
    'huggingface', 'transformers', 'langchain', 'fastai',
    'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'redis',
    'cassandra', 'elasticsearch', 'hbase', 'dynamodb',
    'data analysis', 'data visualization', 'data wrangling', 'data cleaning',
    'exploratory data analysis', 'eda', 'etl', 'data pipeline',
    'statistics', 'probability', 'linear algebra',
    'tableau', 'power bi', 'excel', 'google analytics', 'looker',
    'jupyter', 'google colab',
    'spark', 'hadoop', 'hive', 'kafka', 'airflow', 'databricks', 'snowflake',
    'react', 'node', 'nodejs', 'html', 'css', 'flask', 'django', 'fastapi',
    'rest api', 'graphql', 'spring', 'express',
    'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes',
    'ci/cd', 'jenkins', 'terraform', 'linux', 'git', 'github', 'mlops',
    'llm', 'gpt', 'bert', 'distilbert', 'prompt engineering', 'fine tuning',
    'rag', 'vector database', 'embeddings', 'openai', 'claude', 'langchain',
    'communication', 'teamwork', 'leadership', 'problem solving',
    'critical thinking', 'project management', 'agile', 'scrum',
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def extract_skills(text):
    text_lower = text.lower()
    found = []
    for skill in SKILLS:
        if len(skill) <= 3:
            # Must be a whole word, not inside another word
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found.append(skill)
        else:
            if skill in text_lower:
                found.append(skill)
    return list(set(found))

def match_resume(resume_text, jd_text):
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)
    
    # TF-IDF score
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

    # BERT score
    embeddings = model.encode([clean_resume, clean_jd])
    bert_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100

    # Weighted final score
    final_score = round((0.3 * tfidf_score) + (0.7 * bert_score), 2)

    # Skills
    resume_skills = extract_skills(clean_resume)
    jd_skills = extract_skills(clean_jd)
    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    return {
        'tfidf_score': round(tfidf_score, 2),
        'bert_score': round(bert_score, 2),
        'final_score': final_score,
        'matched_skills': matched,
        'missing_skills': missing,
    }

# --- UI ---
st.title("🎯 Smart Resume-JD Matcher")
st.markdown("Upload your resume and paste a job description to see how well you match!")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Your Resume")
    upload_option = st.radio("Choose input method:", ["Upload PDF", "Paste Text"])
    
    resume_text = ""
    if upload_option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
        if uploaded_file:
            resume_text = extract_text_from_pdf(uploaded_file)
            st.success("✅ Resume uploaded successfully!")
    else:
        resume_text = st.text_area("Paste your resume text here:", height=300)

with col2:
    st.subheader("💼 Job Description")
    jd_text = st.text_area("Paste the job description here:", height=350)

st.divider()

if st.button("Analyze Match", use_container_width=True):
    if not resume_text or not jd_text:
        st.error("Please provide both a resume and a job description!")
    else:
        with st.spinner("Analyzing..."):
            result = match_resume(resume_text, jd_text)

        st.subheader("📊 Match Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Final Score", f"{result['final_score']}%")
        col2.metric("📐 TF-IDF Score", f"{result['tfidf_score']}%")
        col3.metric("🧠 BERT Score", f"{result['bert_score']}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Matched Skills")
            if result['matched_skills']:
                for skill in sorted(result['matched_skills']):
                    st.success(skill)
            else:
                st.warning("No matching skills found")

        with col2:
            st.subheader("❌ Missing Skills")
            if result['missing_skills']:
                for skill in sorted(result['missing_skills']):
                    st.error(skill)
            elif result['matched_skills']:
                st.success("You have all the required skills! 🎉")
            else:
                st.warning("No skills detected in job description")

        st.divider()

        st.subheader("💡 Recommendation")
        score = result['final_score']
        if score >= 40:
            st.success("🟢 Strong match! You should apply for this role.")
        elif score >= 25:
            st.warning("🟡 Moderate match. Consider upskilling in the missing areas before applying.")
        else:
            st.error("🔴 Low match. This role may not align well with your current profile.")