from pymilvus import connections, Collection

# Milvus 서버 연결
connections.connect("default", host="localhost", port="19530")

# 확인할 컬렉션 이름
collection_name = "spotify_music_content"

# 컬렉션 인스턴스 생성
collection = Collection(collection_name)

# 컬렉션 정보 출력
print(f"\n📂 컬렉션 이름: {collection.name}")
print(f"📝 설명: {collection.description}")
print(f"📊 필드 스키마:")
for field in collection.schema.fields:
    print(f" - {field.name} ({field.dtype})")

# 컬렉션 로드
collection.load()

# 총 레코드 수 출력
print(f"\n📦 총 레코드 수: {collection.num_entities}")

# 앞에서부터 5개 데이터만 샘플로 확인
print("\n🧪 샘플 데이터 (앞에서 5개):")
results = collection.query(
    expr="",
    output_fields=["id", "name", "description"],
    limit=5
)
for i, result in enumerate(results, 1):
    print(f"\n#{i}")
    for k, v in result.items():
        print(f"{k}: {v}")