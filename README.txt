1. Cài môi trường:
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\activate
pip install -r requirements.txt

2. Tạo file .env
#Mở file .env và thêm API Key của Gemini.
Ví dụ: GOOGLE_API_KEY="Key của bạn"

3. Chạy chương trình
Mở Terminal và chạy lệnh:
streamlit run app.py
