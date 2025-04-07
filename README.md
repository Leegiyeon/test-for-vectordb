# 🎵 Emotion-Based Music Recommendation with Milvus & Transformers

이 프로젝트는 사용자의 입력 문장에서 감정을 분석하고, 해당 감정에 맞는 음악을 Milvus 벡터 검색을 통해 추천하는 시스템입니다.

---

## 📌 요약

- **감정 분석 모델**: `nateraw/bert-base-uncased-emotion` (HuggingFace)
- **문장 임베딩 모델**: `intfloat/multilingual-e5-base` (SentenceTransformers)
- **벡터 검색 엔진**: [Milvus v2.3.9](https://milvus.io)
- **벡터 DB 연동**: pymilvus
- **스토리지**: MinIO
- **메타 저장소**: etcd
- **추천 로직**: 사용자 입력 → 감정 분류 → 관련 키워드 변환 → 임베딩 → Milvus 검색

---

## 🏗️ 프로젝트 구조

```
test-for-vectordb/
├── app/
│   ├── api/                 # API 라우터 (미구현 또는 향후 확장)
│   ├── db/                  # Milvus 관련 데이터 처리 스크립트
│   ├── services/            # 추천 시스템 핵심 로직
│   └── test_recommender.py  # 테스트 스크립트
├── config/                  # Milvus 설정
├── docker/                  # docker-compose 설정
├── etcd/                    # etcd 실행 파일 (etcdctl 포함)
├── main.py                  # 프로젝트 진입점 (필요 시 사용)
├── .env                     # 환경변수 설정 파일
├── requirements.txt         # Python 의존성
└── README.md                # 설명서
```

---

## 🚀 실행 방법

### 1. 도커 서비스 실행 (Milvus + etcd + MinIO)

```bash
cd docker
docker-compose up -d
```

### 2. 가상환경 설정 및 패키지 설치

```bash
conda activate prompt-env
pip install -r requirements.txt
```

### 3. 컬렉션 및 인덱스 생성

```bash
# (이미 있다면 생략 가능)
python app/db/milvus_schema.py
```

### 4. 샘플 데이터 삽입

```bash
python app/db/insert_sample_data.py
```

### 5. 추천 테스트 실행

```bash
python app/test_recommender.py
```

---

## 🧠 기술 스택

| 구성 요소         | 사용 기술                                 |
|------------------|--------------------------------------------|
| 벡터 검색 DB     | Milvus v2.3.9                              |
| 벡터 인덱싱      | IVF_FLAT + COSINE                          |
| 임베딩 모델      | intfloat/multilingual-e5-base              |
| 감정 분류 모델   | nateraw/bert-base-uncased-emotion          |
| 백엔드 프레임워크| Python + pymilvus + transformers           |
| 스토리지         | MinIO                                      |
| 메타데이터 저장소| etcd                                       |

---

## 🔧 기타 참고

- `docker-compose.yml`에는 Attu 연동을 위한 포트 및 네트워크 설정 포함
- `test_recommender.py`는 감정에 따라 음악을 추천해주는 엔트리 포인트
- 추후 FastAPI 등으로 API 구성도 가능

---

## 💡 향후 확장 방향

- 사용자 로그 기반 개인화
- Spotify API 연동
- 웹 서비스 프론트엔드 추가
- FastAPI 기반 API 서버 구성
