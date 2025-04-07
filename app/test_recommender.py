# test_recommender.py
from services.recommender import recommend_music

if __name__ == "__main__":
    user_input = "요즘 기분이 너무 우울해"  # 테스트 문장
    recommendations = recommend_music(user_input)

    for idx, music in enumerate(recommendations, 1):
        print(f"[{idx}] {music['name']}")
        print(f"  - Description: {music['description']}")
        print(f"  - Image URL: {music['image_url']}")
        print()