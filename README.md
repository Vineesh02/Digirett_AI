Lovdata RAG Application – Run Guide

This repository contains Backend (FastAPI) and Frontend (React).

▶️ Backend – Run Commands
cd backend

python -m venv .venv


Activate virtual environment

Windows

.venv\Scripts\activate


pip install -r requirements.txt

python -m uvicorn app.main:app --reload


Backend runs at:

http://127.0.0.1:8000 

▶️ Frontend – Run Commands

cd frontend

npm install

npm start


Frontend runs at:

http://localhost:3000