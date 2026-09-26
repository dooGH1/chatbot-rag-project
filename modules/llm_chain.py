import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Tải biến môi trường từ file .env ở thư mục gốc dự án
load_dotenv()

def get_rag_chain(vector_db):
    """Khởi tạo LLM và chuỗi truy xuất (Retrieval Chain)"""
    # Lấy API Key từ biến môi trường và truyền tường minh vào lớp ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("Không tìm thấy GOOGLE_API_KEY trong file .env hoặc biến môi trường!")

    # Cấu hình LLM Gemini với api_key tường minh
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1, 
        google_api_key=api_key
    )
    
    # Tạo Prompt chuẩn
    template = """
    Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu được cung cấp.
    Dưới đây là các đoạn thông tin được trích xuất từ tài liệu (Ngữ cảnh):
    
    {context}
    
    Câu hỏi của người dùng: {input}
    
    Yêu cầu:
    1. Chỉ sử dụng thông tin trong phần Ngữ cảnh để trả lời.
    2. Nếu phần Ngữ cảnh không chứa câu trả lời, hãy nói: "Dựa trên tài liệu được cung cấp, tôi không tìm thấy thông tin để trả lời câu hỏi này." Tuyệt đối không tự bịa ra.
    3. Cố gắng trả lời ngắn gọn, dễ hiểu và giữ nguyên các thuật ngữ kỹ thuật.
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Biến FAISS thành một retriever (lấy top 3 đoạn liên quan nhất)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Nối các module lại thành 1 chuỗi xử lý RAG hoàn chỉnh
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain