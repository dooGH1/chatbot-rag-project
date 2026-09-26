import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

def get_rag_chain(vector_db):
    """Khởi tạo LLM và chuỗi truy xuất RAG tương thích với các phiên bản langchain mới"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("Không tìm thấy GOOGLE_API_KEY trong file .env hoặc biến môi trường!")

    # Khởi tạo mô hình Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1, 
        google_api_key=api_key
    )
    
    # Định nghĩa Prompt Template chuẩn
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
    
    # Thiết lập retriever lấy top 3 đoạn liên quan nhất
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Tạo chuỗi xử lý tài liệu và chuỗi retrieval
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain