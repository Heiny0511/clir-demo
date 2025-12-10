import streamlit as st
import pandas as pd
import torch
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load CLIR model
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-m3")
model = load_model()

# 2. Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("clir_dataset.csv")
    return df
df = load_data()

# 3. Encode documents
@st.cache_resource
def encode_docs(docs):
    return model.encode(docs, convert_to_tensor=True, show_progress_bar=True)
doc_embeddings = encode_docs(df["content"].tolist())

# 4. Load Translation Model (NLLB)
@st.cache_resource
def load_translation_models():
    NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
    TARGET_LANG_TOKEN = "vie_Latn"     
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)

    TARGET_LANG_ID = nllb_tokenizer.convert_tokens_to_ids(TARGET_LANG_TOKEN)
    
    return nllb_tokenizer, nllb_model, TARGET_LANG_TOKEN, TARGET_LANG_ID

nllb_tokenizer, nllb_model, TARGET_LANG_TOKEN, TARGET_LANG_ID = load_translation_models() 

def translate_to_vietnamese(text):
    try:
        lang = detect(text)
    except:
        lang = 'en'
        
    lang_mapping = {
        'en': 'eng_Latn',
        'zh': 'zho_Hans',
        'zh-cn': 'zho_Hans',
        'zh-tw': 'zho_Hant',
        'vi': TARGET_LANG_TOKEN 
    }
    
    src_lang_code = lang_mapping.get(lang)

    if src_lang_code is None or src_lang_code == TARGET_LANG_TOKEN:
        return text 
    
    nllb_tokenizer.src_lang = src_lang_code 
    
    inputs = nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    outputs = nllb_model.generate(
        **inputs, 
        forced_bos_token_id=TARGET_LANG_ID, # Dùng ID đã được convert
        max_length=512
    )
    
    translation = nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# 5. Streamlit UI
st.title("WHALEWITHWINGS.COM - TÌM KIẾM ĐA NGÔN NGỮ")
query = st.text_input(" Nhập nội dung cần tìm kiếm của bạn:")
if query:
    
    # BƯỚC 1: Xử lý tìm kiếm (Nhanh)
    with st.spinner('Đang tìm kiếm kết quả liên quan...'):
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        top_results = scores.topk(5)
        
        st.subheader("Top kết quả tìm kiếm:")
    
    translation_placeholders = []
    
    # BƯỚC 2: VÒNG LẶP HIỂN THỊ KẾT QUẢ GỐC (Rất nhanh)
    for i, idx in enumerate(list(top_results.indices)):
        idx = int(idx)  
        original_text = df.iloc[idx]['content']
        
        try:
            original_lang = detect(original_text).upper()
        except:
            original_lang = 'UNDEFINED'
        
        # HIỂN THỊ KẾT QUẢ GỐC NGAY LẬP TỨC
        st.markdown(f"### **{df.iloc[idx]['title']}**")
        st.markdown(f"[Link]({df.iloc[idx]['url']})")
        st.write(f"#### Tài liệu gốc ({original_lang}):")
        st.write(original_text)
        
        # Tạo placeholder và lưu thông tin
        placeholder = st.empty()
        # Lưu thông tin cần thiết để truyền vào hàm dịch
        translation_placeholders.append((original_lang, original_text, placeholder))
        
    st.markdown("---")
    
    # BƯỚC 3: VÒNG LẶP XỬ LÝ DỊCH THUẬT (Chậm, nhưng chạy sau)
    for original_lang, original_text, placeholder in translation_placeholders:
        
        # Chỉ dịch nếu không phải là Tiếng Việt
        if original_lang != 'VI':
            
            # Khối này là nơi mất thời gian, nhưng không làm chặn UI ban đầu
            vietnamese_text = translate_to_vietnamese(original_text)
            
            # Cập nhật placeholder
            with placeholder.container():
                st.write("#### 🇻🇳 Dịch sang tiếng Việt:")
                st.success(vietnamese_text)
