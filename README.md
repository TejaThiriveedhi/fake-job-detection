# 🔍 Fake Job Posting Detection System

A machine learning system that classifies job postings as **real or fraudulent** using NLP and structured metadata features. Built with XGBoost and TF-IDF, achieving a **ROC-AUC of 0.9862** and **80% recall on the fake class**.

---

## 📌 Problem Statement

Online job platforms are increasingly targeted by scammers posting fraudulent listings to collect personal data or fees from job seekers. This project builds a binary classifier that flags suspicious postings before they reach applicants — prioritizing **high recall on the fake class** to minimize missed scams.

---

## 📊 Dataset

**Source:** [Kaggle – Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

| Property | Details |
|---|---|
| Total records | 17,880 job postings |
| Features | 18 columns (text + metadata) |
| Target column | `fraudulent` (0 = Real, 1 = Fake) |
| Class distribution | 17,014 Real (95.16%) / 866 Fake (4.84%) |

Key columns used: `title`, `company_profile`, `description`, `requirements`, `benefits`, `telecommuting`, `has_company_logo`, `has_questions`

> The dataset is **highly imbalanced** (~20:1 ratio), addressed via `scale_pos_weight` in XGBoost.

---

## ⚙️ Methodology

### 1. Text Preprocessing
All five text columns (`title`, `company_profile`, `description`, `requirements`, `benefits`) are concatenated into a single `full_text` field, then cleaned:
- Lowercasing
- Removal of URLs, HTML tags, and special characters
- NLTK tokenization
- Stopword removal
- WordNet lemmatization

### 2. Feature Engineering
A `ColumnTransformer` combines two feature types:

| Feature Type | Details |
|---|---|
| TF-IDF (text) | Top 5,000 unigrams + bigrams, `min_df=5`, English stopwords |
| Meta features | `telecommuting`, `has_company_logo`, `has_questions` (passed through directly) |

### 3. Class Imbalance Handling
XGBoost's built-in `scale_pos_weight` parameter is set to ~19.7 (ratio of real to fake samples in training), which re-weights the loss function without generating synthetic samples.

### 4. Models Trained

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline — `class_weight='balanced'`, 1000 max iterations |
| XGBoost | Final model — 200 estimators, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight=~19.7` |

Train/test split: 80/20, stratified on the target label.

---

## 📈 Results

### XGBoost — Final Model

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (Real) | 0.99 | 0.99 | 0.99 | 3403 |
| 1 (Fake) | 0.76 | 0.80 | 0.78 | 173 |

| Metric | Score |
|---|---|
| Overall Accuracy | 98% |
| ROC-AUC | 0.9862 |
| Fake jobs caught | 138 / 173 (80%) |
| Fake jobs missed | 35 |

### Key Insights from Feature Importance

Top indicators of fake postings identified by XGBoost:
- Phrases: "work home", "quality candidate", "finance option", "encouraged", "real estate"
- Structural signal: absence of `has_company_logo`

Word clouds confirm distinct vocabulary between real postings (professional, experience-focused) and fake postings (vague, high-reward language).

---

## 🗂️ Project Structure

```
fake-job-detection/
│
├── data/
│   └── fake_job_postings.csv        # Kaggle dataset (not pushed — see note below)
│
├── notebooks/
│   └── fake_job_detection.ipynb     # Full pipeline: EDA → preprocessing → training → evaluation
│
├── reports/
│   └── class_imbalance_bar.png      # Class distribution visualization
│
├── app.py                           # Streamlit web app for live predictions
├── preprocessor.pkl                 # Saved ColumnTransformer (TF-IDF + meta features)
├── xgb_model.pkl                    # Saved trained XGBoost model
├── requirements.txt                 # Python dependencies
└── README.md
```

> **Note on dataset:** The CSV is not committed due to file size. Download `fake_job_postings.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) and place it in the `data/` folder before running the notebook.

---

## 🖥️ Streamlit App

The app lets you enter job details and get an instant fraud probability score.

**Run locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Input fields:**
- Job title, company profile, description, requirements, benefits
- Whether the role is remote (`telecommuting`)
- Whether the company has a logo (`has_company_logo`)
- Whether the posting includes screening questions (`has_questions`)

**Output:** A risk verdict (✅ Appears Legitimate / 🚨 High Risk – Likely Fake) with a fake probability percentage.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| pandas, numpy | Data manipulation |
| NLTK | Tokenization, stopword removal, lemmatization |
| scikit-learn | TF-IDF, ColumnTransformer, Logistic Regression, metrics |
| XGBoost | Final classification model |
| WordCloud | EDA visualization |
| matplotlib, seaborn | Plots and confusion matrices |
| joblib | Model serialization |
| Streamlit | Web app interface |

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/TejaThiriveedhi/fake-job-detection.git
cd fake-job-detection

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (run once)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Place the dataset
# Download fake_job_postings.csv from Kaggle → put it in data/

# 6. Run the notebook
jupyter notebook notebooks/fake_job_detection.ipynb

# 7. Launch the Streamlit app
streamlit run app.py
```

---

## 🔮 Future Work

- Integrate transformer models (BERT / RoBERTa) for deeper text understanding
- Train on India-specific datasets (Naukri, LinkedIn) to improve regional coverage
- Deploy as a browser extension or mobile app for real-time checking
- Add SHAP explainability to show why a posting was flagged
- Implement continuous learning to adapt to new scam patterns

---

## ⚠️ Disclaimer

This is an academic project built for educational purposes as part of the Data Warehousing and Data Mining (CSE_4060) course. The model should not be used as the sole basis for flagging job postings. Always verify suspicious listings independently.

---

## 👤 Author

**Teja Thiriveedhi**  
[GitHub](https://github.com/TejaThiriveedhi)
