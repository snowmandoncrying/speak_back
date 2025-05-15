from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from app.router.full_router import router as full_router

app = FastAPI()

# CORS 설정 (모든 origin 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(full_router)
# 라우터 등록은 이후에 추가 예정
# from app.routers.router import router
# app.include_router(router)

from app.router.speech_router import router as speech_router
app.include_router(speech_router) 

from app.services.audio_utils import convert_to_wav
from app.services.whisper_service import run_whisper_transcribe
from app.services.filler_llm_detector import analyze_filler_from_text, build_filler_map_from_result 