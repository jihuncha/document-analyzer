from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import PyPDF2
import io

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="Document Analyzer API", version="1.0.0")

# CORS 설정 (모바일 앱에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드 폴더 생성
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# AI 관련 초기화
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# 요청 모델
class QuestionRequest(BaseModel):
    question: str
    filename: str = None

def extract_text_from_file(file_path: str) -> str:
    """파일에서 텍스트 추출"""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.pdf'):
        try:
            # 방법 1: PyPDF2로 시도
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 0:
                        text += page_text + "\n"
                
                # 텍스트가 너무 짧거나 깨져있으면 다른 방법 시도
                if len(text.strip()) < 50 or any(ord(char) > 65535 for char in text[:100]):
                    raise Exception("Text extraction failed with PyPDF2")
                
                return text
        except:
            # 방법 2: pdfplumber 사용 (더 나은 한글 지원)
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                # pdfplumber가 없으면 에러 메시지와 함께 원본 텍스트 반환
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    return f"PDF 텍스트 추출에 문제가 있을 수 있습니다. 원본 텍스트: {text}"
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text: str, chunk_size: int = 500) -> list:
    """텍스트를 청크로 분할"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

@app.get("/")
async def root():
    return {"message": "Document Analyzer API is running!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """파일 업로드 및 벡터 DB에 저장"""
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 텍스트 추출
        text = extract_text_from_file(file_path)
        
        # 텍스트 청킹
        chunks = chunk_text(text)
        
        # 임베딩 생성 및 벡터 DB에 저장
        embeddings = embedding_model.encode(chunks).tolist()
        
        # 기존 문서 삭제 (같은 파일명이면)
        try:
            collection.delete(where={"filename": file.filename})
        except:
            pass
        
        # 새 문서 추가
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"filename": file.filename, "chunk_id": i}],
                ids=[f"{file.filename}_chunk_{i}"]
            )
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "size": len(content),
            "chunks_created": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.get("/files")
async def list_files():
    """업로드된 파일 목록 조회"""
    files = []
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        file_size = os.path.getsize(file_path)
        files.append({
            "filename": filename,
            "size": file_size
        })
    return {"files": files}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """문서에 대한 질문-답변 API"""
    try:
        # 질문을 임베딩으로 변환
        question_embedding = embedding_model.encode([request.question]).tolist()[0]
        
        # 관련 문서 검색 (상위 3개)
        if request.filename:
            # 파일명이 있으면 해당 파일에서만 검색
            # 먼저 전체 검색 후 수동 필터링
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=10,  # 더 많이 가져와서 필터링
                include=['documents', 'metadatas']
            )
            
            # 수동으로 해당 파일명만 필터링
            filtered_docs = []
            filtered_metas = []
            
            if results['documents'] and results['documents'][0]:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    if meta.get('filename') == request.filename:
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                        if len(filtered_docs) >= 3:  # 상위 3개만
                            break
            
            # 결과 재구성
            if filtered_docs:
                results = {
                    'documents': [filtered_docs],
                    'metadatas': [filtered_metas]
                }
            else:
                results = {'documents': [[]], 'metadatas': [[]]}
        else:
            # 파일명이 없으면 전체 검색
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=3,
                include=['documents', 'metadatas']
            )
        
        if not results['documents'][0]:
            return {
                "answer": "관련된 문서 내용을 찾을 수 없습니다.", 
                "sources": [],
                "searched_filename": request.filename
            }
        
        # 검색된 문서들을 컨텍스트로 구성
        context = "\n\n".join(results['documents'][0])
        
        # Claude에게 질문
        prompt = f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.

문서 내용:
{context}

질문: {request.question}

답변:"""

        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "answer": response.content[0].text,
            "sources": results['metadatas'][0],
            "context_used": len(results['documents'][0]),
            "searched_filename": request.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

@app.get("/debug/all")
async def debug_all_files():
    """저장된 모든 데이터 확인"""
    try:
        all_data = collection.get()
        return {
            "total_chunks": len(all_data['documents']) if all_data['documents'] else 0,
            "all_filenames": list(set([meta.get('filename', 'no filename') for meta in all_data['metadatas']])) if all_data['metadatas'] else [],
            "sample_data": [
                {
                    "id": all_data['ids'][i] if all_data['ids'] else 'no id',
                    "filename": all_data['metadatas'][i].get('filename', 'no filename') if all_data['metadatas'] else 'no metadata',
                    "chunk_id": all_data['metadatas'][i].get('chunk_id', 'no chunk_id') if all_data['metadatas'] else 'no metadata',
                    "document_preview": all_data['documents'][i][:100] + "..." if all_data['documents'] and len(all_data['documents'][i]) > 100 else all_data['documents'][i] if all_data['documents'] else 'no document'
                }
                for i in range(min(3, len(all_data['documents']) if all_data['documents'] else 0))
            ]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)