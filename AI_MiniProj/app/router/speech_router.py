from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.audio_utils import convert_to_wav
from app.services.whisper_service import run_whisper_transcribe
from app.services.filler_llm_detector import analyze_filler_from_text, build_filler_map_from_result
import os

router = APIRouter(prefix="/api/speech", tags=["Speech Analysis"])

@router.post("/analyze")
def analyze_speech(file: UploadFile = File(...)):
    """
    업로드된 음성 파일을 받아서, whisper 및 추가 분석을 수행하고,
    문장 단위로 통합된 결과를 반환합니다.
    """
    try:
        # 1. 파일을 .wav로 변환
        wav_path = convert_to_wav(file)

        # 2. Whisper 한 번만 실행
        whisper_result = run_whisper_transcribe(wav_path)
        segments = whisper_result.get("segments", [])
        full_text = whisper_result.get("text", "")

        # 3. 말버릇(LLM) 분석 (Whisper 텍스트만 사용)
        filler_result = analyze_filler_from_text(full_text)
        # segment.id별로 말버릇 매핑
        filler_map = build_filler_map_from_result(filler_result, segments)

        # 4. (추가 분석 서비스: 발음, 속도, 억양 등 segment별로 결과 추가)
        for seg in segments:
            seg_id = seg.get("id")
            # 예시: seg["pronunciation"] = analyze_pronunciation_for_segment(wav_path, seg["start"], seg["end"])
            # 예시: seg["speed"] = analyze_speed_for_segment(wav_path, seg["start"], seg["end"])
            # 예시: seg["intonation"] = analyze_intonation_for_segment(wav_path, seg["start"], seg["end"])
            # 말버릇 결과는 filler_map에서 가져와서 추가
            seg["filler"] = filler_map.get(seg_id, "없음")
            # 기타 분석 결과도 필요시 추가

        # 5. 통합 결과 생성
        merged_segments = []
        for seg in segments:
            merged = {
                "startPoint": seg.get("start"),
                "endPoint": seg.get("end"),
                "word": seg.get("text"),
                "speed": seg.get("speed"),
                "volume": seg.get("volume"),
                "intonation": seg.get("intonation"),
                "pronunciation": seg.get("pronunciation"),
                "filler": seg.get("filler"),
                "silence": seg.get("silence"),
            }
            merged_segments.append(merged)

        # 임시 파일 삭제
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return JSONResponse(content={"segments": merged_segments})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 