# app.py - Fake Job Posting Detector (Streamlit)

import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ─── Load saved models ───
try:
    preprocessor = joblib.load('preprocessor.pkl')
    xgb = joblib.load('xgb_model.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please place 'preprocessor.pkl' and 'xgb_model.pkl' in the same folder as this app.")
    st.stop()

# ─── Text cleaning ───
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# ─── Streamlit UI ───
st.set_page_config(page_title="Fake Job Detector", page_icon="🔍", layout="centered")

st.title("🔍 Fake Job Posting Detector")
st.markdown("Detect potentially fraudulent job postings using machine learning (XGBoost + TF-IDF)")

with st.form("job_input"):
    st.subheader("Enter Job Details")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Job Title", "Work From Home - Data Entry")
        company_profile = st.text_area("Company Profile", "", height=80)
        telecommuting = st.checkbox("Offers Work from Home?", value=True)

    with col2:
        description = st.text_area("Job Description", "Earn ₹40,000–₹80,000 per month. No experience required.", height=140)
        requirements = st.text_input("Requirements", "Basic typing skills")
        benefits = st.text_input("Benefits", "Daily payment, flexible hours")

    col3, col4 = st.columns(2)
    with col3:
        has_logo = st.radio("Has Company Logo?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with col4:
        has_questions = st.radio("Has Application Questions?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Analyze Posting", type="primary")

if submitted:
    with st.spinner("Analyzing..."):
        full_text = f"{title} {company_profile} {description} {requirements} {benefits}"
        cleaned = clean_text(full_text)

        input_data = pd.DataFrame({
            'clean_text': [cleaned],
            'telecommuting': [int(telecommuting)],
            'has_company_logo': [has_logo],
            'has_questions': [has_questions]
        })

        transformed = preprocessor.transform(input_data)
        prob = xgb.predict_proba(transformed)[0][1]

        if prob > 0.5:
            st.error(f"🚨 **HIGH RISK – Likely FAKE**  \nFake probability: **{prob:.1%}**")
        else:
            st.success(f"✅ **Appears LEGITIMATE**  \nFake probability: **{prob:.1%}**")

        st.divider()
        st.caption("Note: This is an educational model. Always verify suspicious offers independently.")