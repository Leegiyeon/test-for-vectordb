import unittest
from app.services.recommender import recommend_music

class TestRecommenderKorean(unittest.TestCase):

    def test_joy_emotion(self):
        result = recommend_music("오늘 너무 행복해!")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        print("\n[기쁨] 추천 결과:")
        for music in result:
            print(f"- {music['name']} ({music['id']}) - 감정 태그: {music.get('emotion', 'N/A')}")

    def test_sadness_emotion(self):
        result = recommend_music("정말 우울하고 눈물이 나.")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        print("\n[슬픔] 추천 결과:")
        for music in result:
            print(f"- {music['name']} ({music['id']}) - 감정 태그: {music.get('emotion', 'N/A')}")

    def test_unknown_emotion(self):
        result = recommend_music("요즘 날씨가 오락가락하네.")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        print("\n[기타 감정] 추천 결과:")
        for music in result:
            print(f"- {music['name']} ({music['id']}) - 감정 태그: {music.get('emotion', 'N/A')}")

if __name__ == '__main__':
    unittest.main()
