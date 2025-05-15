from app.services.intonation_analyzer import analyze_intonation
from app.services.speed_analyzer import analyze_speed

def analyze_full_from_segments(audio_path: str, segments, text):
    # 억양 분석
    intonation_results, avg_pitch_std, pitch_ranges = analyze_intonation(audio_path, segments)
    # pitch 검출률 계산
    pitch_coverages = [r["pitch_coverage"] for r in intonation_results if r.get("pitch_coverage") is not None]
    pitch_detection_rate = round(sum(pitch_coverages) / len(pitch_coverages), 2) if pitch_coverages else 0.0
    # 속도 분석
    speed_results, avg_spm, avg_wpm = analyze_speed(audio_path, segments)
    # 문장별 피드백 병합
    feedback_by_sentence = []
    for i, seg in enumerate(segments):
        start = round(seg["start"], 2)
        end = round(seg["end"], 2)
        sentence_text = seg.get("text") if seg.get("text") else "[문장 텍스트 없음]"
        intonation_feedback = intonation_results[i]["intonation_feedback"] if i < len(intonation_results) else None
        speed_feedback = speed_results[i]["feedback"] if i < len(speed_results) else None
        feedback_by_sentence.append({
            "start_point": start,
            "end_point": end,
            "word": sentence_text,
            "speed": speed_feedback,
            "intonation": intonation_feedback
        })
    return {
        "text": text,
        "feedback_by_sentence": feedback_by_sentence,
        "average_spm": avg_spm,
        "average_wpm": avg_wpm,
        "average_pitch_std": avg_pitch_std,
        "pitch_detection_rate": pitch_detection_rate
    } 