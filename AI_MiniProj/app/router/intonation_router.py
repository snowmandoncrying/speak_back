from fastapi import APIRouter, UploadFile, File
from app.services.intonation_service import analyze_intonation_from_audio, get_pretty_intonation_result
import tempfile
import shutil
from fastapi.responses import Response

router = APIRouter()

@router.post("/intonation/analyze")
async def analyze_intonation_api(file: UploadFile = File(...)):
    # 업로드 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    # 분석 함수 호출
    result = analyze_intonation_from_audio(tmp_path)
    formatted_str = get_pretty_intonation_result(result)
    return Response(content=formatted_str, media_type="text/plain") 