from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router.intonation_router import router as intonation_router

app = FastAPI()

# CORS 설정 (모든 origin 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(intonation_router)
# 라우터 등록은 이후에 추가 예정
# from app.routers.router import router
# app.include_router(router) 