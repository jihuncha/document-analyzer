# Document Analyzer API

AI 기반 문서 분석 및 질문-답변 서비스 (Use Venv)

## 기능
- PDF/TXT 파일 업로드
- 문서 내용 자동 분석 및 임베딩
- 자연어 질문으로 문서 내용 검색
- Claude API 기반 답변 생성

## 설치 및 실행

### 1. 프로젝트 클론
```bash
git clone <repository-url>
cd document-analyzer
```

### 2. 가상환경 설정
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정
`.env` 파일 생성 후 API 키 입력:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 5. 서버 실행
```bash
python3 main.py
```

API 문서: http://localhost:8000/docs

## 사용법

1. `/upload` - 문서 업로드
2. `/ask` - 문서에 질문하기
3. `/files` - 업로드된 파일 목록

## 기술 스택
- FastAPI
- Claude 3.5 Sonnet
- ChromaDB
- sentence-transformers
