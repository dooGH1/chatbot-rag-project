from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

def create_vector_db(documents):
    """Chia nhỏ văn bản và tạo cơ sở dữ liệu vector FAISS dùng mô hình HuggingFace cục bộ"""
    if not documents:
        st.error("Tài liệu tải lên không chứa văn bản hợp lệ hoặc không đọc được nội dung!")
        return None
        
    # Chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        st.error("Không thể trích xuất đoạn văn bản nào từ tài liệu này.")
        return None
        
    # Sử dụng mô hình embedding đa ngôn ngữ chạy cục bộ (nhẹ, chuẩn, không lỗi API Google)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Tạo Vector Store với FAISS
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db