from fastapi import APIRouter
from app.services.recommender import recommend_music

router = APIRouter()

@router.get("/recommend")
def recommend(q: str):
    return recommend_music(q)
