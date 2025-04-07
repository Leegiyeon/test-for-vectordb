# 🎵 감정 기반 음악 추천 시스템 (Milvus + Transformers)

사용자의 문장에서 감정을 발견하고, 해당 감정에 여우리는 음악을 Milvus 벡터 검색으로 추천하는 Python 기반 프로젝트입니다.

---

## 📌 요약

- **감정 발생 모델**: `alsgyu/sentiment-analysis-fine-tuned-model` (HuggingFace)
- **문장 임벤딩 모델**: `jhgan/ko-sbert-nli` (SentenceTransformers)
- **벡터 검색 엔진**: [Milvus v2.3.9](https://milvus.io)
- **벡터 DB 연동**: pymilvus
- **스토리지**: MinIO
- **메타 저장소**: etcd
- **추천 로직**: 사용자 입력 → 감정 분류 → 관련 키워드 생성 → 문장 임벤딩 → Milvus 검색 결과 기반 추천

---

## 🏗️ 프로젝트 구조

```
test-for-vectordb
├─ README.md
├─ app
│  ├─ api
│  │  ├─ __init__.py
│  │  └─ routes.py
│  ├─ db
│  │  ├─ __init__.py
│  │  ├─ check_collection.py
│  │  ├─ insert_sample_data.py
│  │  └─ milvus_schema.py
│  └─ services
│     ├─ __init__.py
│     └─ recommender.py
├─ config
│  └─ milvus_config.py
├─ docker
│  └─ docker-compose.yml
├─ main.py
├─ requirements.txt
└─ tests
   ├─ __init__.py
   └─ test_recommender.py

```
---

## 🚀 실행 방법

### 1. 도커 환경에서 Milvus + MinIO + etcd 실행

```bash
cd docker
docker compose up -d
cd ..
```

### 2. Python 환경 준비 및 의존성 설치

```bash
conda activate prompt-env  # 또는 python -m venv venv
pip install --upgrade pip
pip install -r requirements.txt
pip install sentence-transformers==2.2.2 transformers==4.30.2 huggingface_hub==0.16.4
```

### 3. Milvus 컬렉션 및 인덱스 생성

```bash
python app/db/milvus_schema.py
```

### 4. 샘플 음악 데이터 생성

```bash
python app/db/insert_sample_data.py
```

### 5. 추천 기능 테스트

```bash
PYTHONPATH=. python tests/test_recommender.py
```

---

## 🧪 GitHub Actions CI 설정

- `docker-compose`로 Milvus 전체 스택 실행
- `milvus_schema.py`로 컬렉션 생성
- `insert_sample_data.py`로 데이터 삽입
- `test_recommender.py`로 테스트 수행
- `transformers`, `sentence-transformers`, `huggingface_hub` 버전 고정으로 CI 에러 해결

CI는 `pull_request` 이벤트에서 자동으로 동작합니다.

---

## 🤠 기술 스택

| 구성 요소         | 사용 기술                                 |
|------------------|--------------------------------------------|
| 벡터 검색 DB     | Milvus v2.3.9                              |
| 벡터 인데스      | IVF_FLAT + COSINE                          |
| 임벤딩 모델      | jhgan/ko-sbert-nli              |
| 감정 분류 모델   | alsgyu/sentiment-analysis-fine-tuned-model  |
| 백업데 로직      | Python + pymilvus + sentence-transformers  |
| 스토리지         | MinIO                                      |
| 메타 저장소      | etcd                                       |
| 테스트           | unittest + GitHub Actions                  |

---

## 🗂️ 참고

- Milvus는 로컬에서 Docker로 구동되며, CI에서는 GitHub Actions 내에서 자동 실행됩니다.

