from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Milvus 연결
connections.connect("default", host="localhost", port="19530")
collection = Collection("spotify_playlists")

# 벡터 임베딩 모델
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")

# 감정 분류 모델
emotion_classifier = pipeline("text-classification", model="jhgan/ko-sroberta-multitask")

# 예시 데이터 10개
playlist_data = [
    {
        "id": "playlist_001",
        "name": "비도 오고 그래서",
        "description": "비가 오는 날, 이별의 감정을 담은 감성 발라드",
        "image_url": "https://example.com/image1.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_001",
        "owner_name": "헤이즈",
        "followers": 90000,
    },
    {
        "id": "playlist_002",
        "name": "좋은 날",
        "description": "맑은 날씨처럼 기분 좋은 에너지를 주는 노래",
        "image_url": "https://example.com/image2.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_002",
        "owner_name": "아이유",
        "followers": 120000,
    },
    {
        "id": "playlist_003",
        "name": "Love Poem",
        "description": "사랑하는 사람에게 바치는 따뜻한 발라드",
        "image_url": "https://example.com/image3.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_003",
        "owner_name": "아이유",
        "followers": 110000,
    },
    {
        "id": "playlist_004",
        "name": "Dynamite",
        "description": "신나는 디스코 팝, 에너지 넘치는 댄스곡",
        "image_url": "https://example.com/image4.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_004",
        "owner_name": "BTS",
        "followers": 200000,
    },
    {
        "id": "playlist_005",
        "name": "너를 만나",
        "description": "운명 같은 사랑을 표현한 고백송",
        "image_url": "https://example.com/image5.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_005",
        "owner_name": "폴킴",
        "followers": 95000,
    },
    {
        "id": "playlist_006",
        "name": "사건의 지평선",
        "description": "이별의 감정을 담은 강렬한 발라드",
        "image_url": "https://example.com/image6.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_006",
        "owner_name": "윤하",
        "followers": 89000,
    },
    {
        "id": "playlist_007",
        "name": "에잇",
        "description": "젊음과 이별을 노래한 감성 팝",
        "image_url": "https://example.com/image7.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_007",
        "owner_name": "아이유 & 슈가",
        "followers": 125000,
    },
    {
        "id": "playlist_008",
        "name": "Spring Day",
        "description": "그리움을 담은 따뜻한 봄 노래",
        "image_url": "https://example.com/image8.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_008",
        "owner_name": "BTS",
        "followers": 210000,
    },
    {
        "id": "playlist_009",
        "name": "Love Dive",
        "description": "사랑의 매력에 빠지는 감정을 표현한 걸그룹 히트곡",
        "image_url": "https://example.com/image9.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_009",
        "owner_name": "IVE",
        "followers": 97000,
    },
    {
        "id": "playlist_010",
        "name": "모든 날, 모든 순간",
        "description": "매일을 특별하게 만들어주는 사랑의 노래",
        "image_url": "https://example.com/image10.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_010",
        "owner_name": "폴킴",
        "followers": 100000,
    }
]

# 필드 수에 맞게 초기화 (12개)
insert_data = [[] for _ in range(12)]

for item in playlist_data:
    # 감정 분류
    emotion_result = emotion_classifier(item["description"])
    tag_emotion = emotion_result[0]['label'] if emotion_result else "N/A"

    # 벡터 임베딩용 텍스트 생성
    track_summary = f"{item['name']} : {item['description']}"
    vector = embedding_model.encode(track_summary, normalize_embeddings=True)

    # 각 필드에 맞춰 데이터 삽입
    insert_data[0].append(item["id"])
    insert_data[1].append(item["name"])
    insert_data[2].append(item["description"])
    insert_data[3].append(item["image_url"])
    insert_data[4].append(item["image_width"])
    insert_data[5].append(item["image_height"])
    insert_data[6].append(item["owner_id"])
    insert_data[7].append(item["owner_name"])
    insert_data[8].append(item["followers"])
    insert_data[9].append(track_summary)
    insert_data[10].append(tag_emotion)
    insert_data[11].append(vector.tolist())

collection.insert(insert_data)
print("✅ 샘플 데이터가 성공적으로 삽입되었습니다.")
