from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.audio_utils import convert_to_wav
# 아래 두 함수는 실제 구현된 서비스에서 import한다고 가정합니다.
from services.test_pronunciation import run_basic_whisper
from services.filler_llm_detector import analyze_filler_with_llm
import os

router = APIRouter(prefix="/api/speech", tags=["Speech Analysis"])

@router.post("/analyze")
def analyze_speech(file: UploadFile = File(...)):
    """
    업로드된 음성 파일을 받아서, whisper 및 LLM 기반 분석을 수행하고,
    문장 단위로 통합된 결과를 반환합니다.
    """
    try:
        # 1. 파일을 .wav로 변환
        wav_path = convert_to_wav(file)

        # 2. 기본 Whisper 실행 (팀원 분석용)
        basic_result = run_basic_whisper(wav_path)  # 예: {"segments": [{"id": 0, ...}, ...]}
        segments = basic_result.get("segments", [])

        # 3. 프롬프트 Whisper+LLM 실행 (말버릇 등)
        filler_result = analyze_filler_with_llm(wav_path)  # 예: {"segments": [{"id": 0, "filler": ...}, ...]}
        filler_map = {seg["id"]: seg for seg in filler_result.get("segments", [])}

        # 4. (추가 분석 서비스 결과 병합은 추후 구현)
        # 예시: speed, volumn, intonation, pronunciation, silence 등
        # 각 서비스별로 segment.id 기준으로 결과를 병합해야 함

        # 5. 통합 결과 생성
        merged_segments = []
        for seg in segments:
            seg_id = seg.get("id")
            merged = {
                "startPoint": seg.get("start"),
                "endPoint": seg.get("end"),
                "word": seg.get("text"),
                # 아래 값들은 추후 실제 분석 결과로 대체
                "speed": "미정",
                "volume": "미정",
                "intonation": "미정",
                "pronunciation": "미정",
                "filler": filler_map.get(seg_id, {}).get("filler", "없음"),
                "silence": seg.get("silence", "미정"),
            }
            merged_segments.append(merged)

        # 임시 파일 삭제
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return JSONResponse(content={"segments": merged_segments})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 