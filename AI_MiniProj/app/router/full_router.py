from fastapi import APIRouter, UploadFile, File
from app.services.full_analysis_service import analyze_full_from_audio
import tempfile
import shutil

router = APIRouter()

@router.post("/analyze/full")
async def analyze_full_api(audio_file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        shutil.copyfileobj(audio_file.file, tmp)
        tmp_path = tmp.name
    result = analyze_full_from_audio(tmp_path)
    return result 