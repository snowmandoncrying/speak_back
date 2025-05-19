from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback
# from app.router.full_router import router as full_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Speech Analysis API", 
            "endpoints": {
                "analyze_speech": "/api/speech/analyze"
            }}

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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("==== FastAPI Global Exception Handler ====")
    print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    ) 