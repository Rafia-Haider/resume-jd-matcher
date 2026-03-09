# 🎯 Smart Resume-JD Matcher

An NLP-powered web app that analyzes how well a resume matches a job description — using a hybrid TF-IDF + DistilBERT sentence embedding approach to compute semantic similarity, extract skills, and generate a gap analysis.

---

## 🚀 Live Demo
https://resume-jd-matcher-xvceypxast37ovonz2kvfz.streamlit.app/

---

## 📌 Features

- **Dual NLP Approach** — combines TF-IDF cosine similarity (keyword matching) with DistilBERT sentence embeddings (semantic understanding) for a more robust match score
- **PDF Resume Upload** — extracts and parses text directly from uploaded PDF resumes
- **Skill Extraction** — identifies 80+ technical and soft skills from both resume and JD using pattern matching with word boundary detection
- **Gap Analysis** — clearly shows matched skills vs missing skills side by side
- **Match Recommendation** — provides actionable feedback based on final score
- **Clean Streamlit UI** — simple, intuitive interface deployable as a web app

---

## 🧠 How It Works

```
Resume (PDF/Text) + Job Description (Text)
            ↓
     Text Cleaning Pipeline
     (lowercase, remove URLs, special chars)
            ↓
   ┌─────────────────────────────┐
   │   Approach 1: TF-IDF        │  → Keyword-level cosine similarity
   │   Approach 2: DistilBERT    │  → Semantic sentence embeddings
   └─────────────────────────────┘
            ↓
   Weighted Score = 0.3 × TF-IDF + 0.7 × BERT
            ↓
   Skill Extraction via pattern matching
            ↓
   Gap Analysis + Recommendation
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| NLP | sentence-transformers, scikit-learn, spaCy |
| ML Model | `all-MiniLM-L6-v2` (DistilBERT-based) |
| Data Processing | pandas, numpy, re |
| Visualization | Streamlit |
| PDF Parsing | PyPDF2 |
| Version Control | Git, GitHub |

---

## 📊 Model Design Decisions

**Why weighted 70% BERT / 30% TF-IDF?**

TF-IDF is precise but purely literal — it only matches exact keywords. BERT understands semantic meaning, so "machine learning engineer" and "ML developer" are recognized as similar even without exact word overlap. The 70/30 weighting favors semantic understanding while still rewarding direct keyword matches that recruiters rely on.

**Why `all-MiniLM-L6-v2`?**

It's a distilled sentence transformer that produces high-quality embeddings at a fraction of the compute cost of full BERT — ideal for a responsive web app without GPU requirements.

**Why word boundary regex for skill extraction?**

A naive `if skill in text` approach caused false positives — for example, the skill `r` would match inside words like "your" or "for". Using `\b` word boundary patterns for short skills (≤3 chars) eliminates this.

---

## 📁 Project Structure

```
resume-jd-matcher/
├── app.py                  # Streamlit web app
├── notebooks/
│   └── exploration.ipynb   # EDA, model experimentation, comparison
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Run Locally

```bash
# Clone the repo
git clone https://github.com/Rafia-Haider/resume-jd-matcher.git
cd resume-jd-matcher

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install streamlit sentence-transformers spacy PyPDF2 scikit-learn pandas matplotlib seaborn
python -m spacy download en_core_web_sm

# Run the app
streamlit run app.py
```

---

## 📈 Example Output

| Metric | Score |
|---|---|
| TF-IDF Score | 9.46% |
| BERT Score | 43.12% |
| Final Score | 33.02% |

**Matched Skills:** python, pandas, numpy, tensorflow, git, sql, classification, regression, nlp, matplotlib, seaborn

**Missing Skills:** pytorch, deep learning, docker, huggingface, statistics, aws

---

## 🔮 Future Improvements

- Add named entity recognition (spaCy NER) for richer skill extraction
- Fine-tune the embedding model on domain-specific resume/JD pairs
- Add resume scoring history across multiple JDs
- Export match report as PDF

---

## 👩‍💻 Author

**Rafia Haider** — CS Student, Bahria University Karachi  