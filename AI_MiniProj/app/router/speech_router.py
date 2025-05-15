from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.audio_utils import convert_to_wav
from app.services.whisper_service import run_whisper_transcribe
from app.services.filler_llm_detector import analyze_filler_from_text, build_filler_map_from_result
from app.services.speed_analyzer import analyze_speed
from app.services.intonation_analyzer import analyze_intonation
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

        # 4. (추가 분석 서비스: 속도, 억양 등 segment별로 결과 추가)
        speed_results, avg_spm, avg_wpm = analyze_speed(wav_path, segments)
        intonation_results, avg_pitch_std, pitch_ranges = analyze_intonation(wav_path, segments)
        for i, seg in enumerate(segments):
            seg_id = seg.get("id")
            seg["filler"] = filler_map.get(seg_id, "없음")
            seg["speed"] = speed_results[i]["feedback"] if i < len(speed_results) else None
            seg["intonation"] = intonation_results[i]["intonation_feedback"] if i < len(intonation_results) else None

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