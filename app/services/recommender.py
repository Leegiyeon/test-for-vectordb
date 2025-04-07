from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections

# Milvus 연결
connections.connect("default", host="localhost", port="19530")

# 감정 분류 모델 (한국어 대응 멀티태스크 모델)
emotion_classifier = pipeline("text-classification", model="alsgyu/sentiment-analysis-fine-tuned-model")

# 벡터 임베딩 모델
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")

# Milvus 컬렉션 로드
collection = Collection("spotify_playlists")
collection.load()

def recommend_music(user_input: str):
    # 감정 분석
    emotion_result = emotion_classifier(user_input)
    print(f"[감정 분석 원본 출력]: {emotion_result}")  # 디버깅 출력

    # 감정 라벨 처리
    emotion_label_raw = emotion_result[0]['label'] if emotion_result else "기타"
    emotion_label = emotion_label_raw.replace("emotion_", "") if "emotion_" in emotion_label_raw else emotion_label_raw

    # 감정 기반 쿼리 강화 문장
    emotion_queries = {
        "슬픔": "눈물이 나는 이별 발라드",
        "기쁨": "신나는 케이팝 댄스곡",
        "화남": "마음을 차분하게 가라앉히는 연주곡",
        "공포": "편안한 분위기의 치유 음악",
        "사랑": "설레는 사랑 노래",
        "놀람": "기분 전환되는 밝은 음악",
        "기타": "감정을 위로하는 음악"
    }

    # 쿼리 문장 생성
    query_text = f"{emotion_queries.get(emotion_label, '')} {user_input}"
    query_vector = embedding_model.encode(query_text, normalize_embeddings=True)

    # Milvus 검색
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=5,
        output_fields=["id", "name", "description", "image_url"]
    )

    # 디버깅용 출력
    print(f"\n[감지된 감정] {emotion_label}")
    for hit in results[0]:
        print(f"- {hit.entity.get('name')} (score: {hit.distance:.4f})")

    return [
        {
            "id": hit.entity.get("id"),
            "name": hit.entity.get("name"),
            "description": hit.entity.get("description"),
            "image_url": hit.entity.get("image_url")
        }
        for hit in results[0]
    ]