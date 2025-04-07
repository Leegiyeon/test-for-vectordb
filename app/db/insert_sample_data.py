from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

connections.connect("default", host="localhost", port="19530")
collection = Collection("spotify_music_content")

# 벡터 임베딩 모델
model = SentenceTransformer("intfloat/multilingual-e5-base")

# 예시 플레이리스트 데이터
data = [
    {
        "id": "playlist_001",
        "name": "Chill Vibes",
        "description": "Relaxing and soothing music for a calm evening.",
        "image_url": "https://example.com/image1.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_001",
        "owner_name": "Alice",
        "followers": 1280
    },
    {
        "id": "playlist_002",
        "name": "Power Workout",
        "description": "High energy tracks to keep you moving.",
        "image_url": "https://example.com/image2.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_002",
        "owner_name": "Bob",
        "followers": 8450
    },
    {
        "id": "playlist_003",
        "name": "Heartbreak Ballads",
        "description": "Songs for the heartbroken and emotional souls.",
        "image_url": "https://example.com/image3.jpg",
        "image_width": 300,
        "image_height": 300,
        "owner_id": "user_003",
        "owner_name": "Charlie",
        "followers": 4920
    }
]

# 임베딩 및 Milvus에 삽입
insert_data = [[], [], [], [], [], [], [], [], [], []]  # 10개의 필드
for item in data:
    text = f"{item['name']} : {item['description']}"
    vector = model.encode(text, normalize_embeddings=True)

    insert_data[0].append(item["id"])
    insert_data[1].append(item["name"])
    insert_data[2].append(item["description"])
    insert_data[3].append(item["image_url"])
    insert_data[4].append(item["image_width"])
    insert_data[5].append(item["image_height"])
    insert_data[6].append(item["owner_id"])
    insert_data[7].append(item["owner_name"])
    insert_data[8].append(item["followers"])
    insert_data[9].append(vector.tolist())

collection.insert(insert_data)
print("Sample data inserted.")
