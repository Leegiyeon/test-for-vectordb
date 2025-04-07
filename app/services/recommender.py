from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from pymilvus import connections
import os

connections.connect("default", host="localhost", port="19530")

# 감정 분류 모델 로드
emotion_classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

# 벡터 임베딩 모델 로드
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

# Milvus 컬렉션 로드
collection = Collection("spotify_music_content")
collection.load()

def recommend_music(user_input: str):
    emotion_result = emotion_classifier(user_input)
    emotion_label = emotion_result[0]['label']

    query_text = {
        "sadness": "슬플 때 듣기 좋은 음악",
        "joy": "기분 좋을 때 어울리는 음악",
        "anger": "화날 때 진정할 수 있는 음악",
        "fear": "불안할 때 마음을 편하게 해주는 음악",
        "love": "사랑에 빠졌을 때 듣고 싶은 음악",
        "surprise": "놀랐을 때 기분 전환되는 음악"
    }.get(emotion_label, "감정에 맞는 추천 음악")

    query_vector = embedding_model.encode(query_text, normalize_embeddings=True)

    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=5,
        output_fields=["id", "name", "description", "image_url"]
    )

    return [
        {
            "id": hit.entity.get("id"),
            "name": hit.entity.get("name"),
            "description": hit.entity.get("description"),
            "image_url": hit.entity.get("image_url")
        }
        for hit in results[0]
    ]
