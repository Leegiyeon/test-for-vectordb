from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, connections

connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="image_width", dtype=DataType.INT64),
    FieldSchema(name="image_height", dtype=DataType.INT64),
    FieldSchema(name="owner_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="owner_name", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="followers", dtype=DataType.INT64),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]

schema = CollectionSchema(fields, description="Spotify Music Content Collection")
collection = Collection(name="spotify_music_content", schema=schema)

collection.create_index(
    field_name="vector",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)
collection.load()
