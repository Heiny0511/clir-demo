import streamlit as st
import pandas as pd
import torch
import re

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ================================
# 1. PREPROCESSING LAYER
# ================================
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ================================
# 2. ENCODING LAYER (BGE-M3)
# ================================
@st.cache_resource
def load_retrieval_model():
    return SentenceTransformer("BAAI/bge-m3")

retrieval_model = load_retrieval_model()

@st.cache_resource
def encode_documents(docs):
    cleaned = [preprocess_text(d) for d in docs]
    return retrieval_model.encode(cleaned, convert_to_tensor=True, show_progress_bar=True)


# ================================
# 3. TRANSLATION + SUMMARIZATION LAYER
# ================================
@st.cache_resource
def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-vi")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-vi")
    return tokenizer, model

translator_tokenizer, translator_model = load_translation_model()

def translate_to_vietnamese(text):
    inputs = translator_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = translator_model.generate(**inputs, max_length=512)
    translation = translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

@st.cache_resource
def load_summarizer():
    # Dùng mT5 hoặc Bart cho tóm tắt
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # CPU, đổi device=0 nếu GPU
    return summarizer

summarizer = load_summarizer()

def summarize_text(text):
    text = text[:1024]  # giới hạn chiều dài cho summarizer
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']


# ================================
# 4. RETRIEVAL LAYER
# ================================
def retrieve_top_k(query, doc_embeddings, df, k=5):
    q_clean = preprocess_text(query)
    q_emb = retrieval_model.encode(q_clean, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, doc_embeddings)[0]
    top = torch.topk(scores, k)
    return top


# ================================
# LOAD DATASET
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\dell\CLIR_Project\clir_dataset.csv")
    return df

df = load_data()
doc_embeddings = encode_documents(df["content"].tolist())


# ================================
# STREAMLIT UI
# ================================
st.title("🌐 CLIR + Summarization + Vietnamese Translation")
st.write("Search multilingual content, summarize and translate to Vietnamese.")

query = st.text_input("🔍 Enter your query:")

if query:
    results = retrieve_top_k(query, doc_embeddings, df, k=5)

    st.subheader("Top Results:")

    for idx in results.indices:
        idx = int(idx)

        original_text = df.iloc[idx]["content"]
        summary_text = summarize_text(original_text)
        vietnamese_text = translate_to_vietnamese(summary_text)

        st.markdown(f"### 📘 {df.iloc[idx]['title']}")
        st.markdown(f"**🔗 Link:** {df.iloc[idx]['url']}")

        st.write("#### 📄 Original Content:")
        st.write(original_text[:400] + "...")

        st.write("#### 📝 Summary:")
        st.info(summary_text)

        st.write("#### 🇻🇳 Translated Summary:")
        st.success(vietnamese_text)

        st.write("---")
