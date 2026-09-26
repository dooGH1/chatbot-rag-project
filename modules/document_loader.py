import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def save_uploaded_file(uploaded_file):
    """Lưu file tải lên từ Streamlit vào thư mục data/"""
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_documents(file_paths):
    """Đọc nội dung các file và trả về danh sách Document"""
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            continue
        documents.extend(loader.load())
    return documents