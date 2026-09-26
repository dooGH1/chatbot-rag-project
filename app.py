import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from modules.vector_db import create_vector_db
from modules.llm_chain import get_rag_chain

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Chatbot RAG Đồ Án Thực Tập Tốt Nghiệp",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Trợ lý AI - Hỏi đáp tài liệu thông minh (RAG)")
st.markdown("Hệ thống hỗ trợ trích xuất thông tin và trả lời câu hỏi dựa trên tài liệu cá nhân của bạn.")

# --- SIDEBAR: Quản lý tài liệu & Cấu hình ---
with st.sidebar:
    st.header("📁 Quản lý Tài liệu")
    
    # Bổ sung tham số accept_multiple_files=True để nhận một hoặc nhiều file
    uploaded_files = st.file_uploader(
        "Tải lên tài liệu (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    # Nút bấm "Xử lý tài liệu" theo đúng yêu cầu
    process_button = st.button("🚀 Xử lý tài liệu")
    
    if process_button:
        if not uploaded_files:
            st.warning("Vui lòng tải lên ít nhất một file tài liệu trước khi bấm xử lý!")
        else:
            os.makedirs("data", exist_ok=True)
            all_documents = []
            
            with st.spinner("Đang đọc và xử lý các tài liệu..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("data", uploaded_file.name)
                    
                    # Bước 2: Lưu từng file vào thư mục data/
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # Bước 3: Đọc nội dung file thành danh sách Document
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        all_documents.extend(loader.load())
                    elif uploaded_file.name.endswith(".docx"):
                        loader = Docx2txtLoader(file_path)
                        all_documents.extend(loader.load())
                    elif uploaded_file.name.endswith(".txt"):
                        loader = TextLoader(file_path, encoding="utf-8")
                        all_documents.extend(loader.load())
            
            if all_documents:
                with st.spinner("Đang chia nhỏ văn bản và tạo Vector Database..."):
                    # Bước 4 & 5: Chia nhỏ, tạo vector database và lưu vào session_state
                    vector_db = create_vector_db(all_documents)
                    if vector_db:
                        st.session_state.vector_db = vector_db
                        st.session_state.rag_chain = get_rag_chain(vector_db)
                        st.session_state.current_files = [f.name for f in uploaded_files]
                        st.session_state.messages = [] # Reset lịch sử chat khi có tài liệu mới
                        
                        # Bước 6: Thông báo hoàn tất xử lý tài liệu
                        st.success(f"Đã xử lý thành công {len(uploaded_files)} tài liệu và sẵn sàng hỏi đáp!")

    st.divider()
    st.markdown("### ℹ️ Thông tin hệ thống")
    st.markdown("- **Mô hình LLM:** Gemini")
    st.markdown("- **Vector Database:** FAISS")
    st.markdown("- **Nhóm thực hiện:** Nhóm 2 PTIT")

# --- MAIN CHAT INTERFACE ---
# Khởi tạo lịch sử chat nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử trò chuyện trên giao diện
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input câu hỏi từ người dùng qua ô chat ở đáy màn hình
if user_query := st.chat_input("Nhập câu hỏi của bạn về tài liệu..."):
    # Kiểm tra xem đã tải tài liệu lên chưa
    if "rag_chain" not in st.session_state:
        st.warning("Vui lòng tải lên tài liệu và bấm nút 'Xử lý tài liệu' ở thanh bên trái trước khi đặt câu hỏi!")
    else:
        # Thêm câu hỏi của người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Gọi RAG Chain để sinh câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm thông tin và tổng hợp câu trả lời..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                except Exception as e:
                    answer = f"Đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
                
                st.markdown(answer)
                # Thêm câu trả lời của trợ lý vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": answer})