import sys
from app.services.stt_service import speech_recognizer
from app.services.intonation_analyzer import analyze_intonation
import platform

FEEDBACK_MAP = {
    "이 문장은 억양 변화 폭이 지나치게 커서 과장된 느낌을 줄 수 있습니다.": "❗ 이 구간은 억양 변화 폭이 다소 커서 과장되게 들릴 수 있습니다. 중요한 내용 외에는 억양을 더 부드럽게 조절해보세요.",
    "이 문장은 발표 전체에 비해 억양이 특히 단조롭게 들릴 수 있습니다.": "❌ 이 구간은 억양 변화가 적어 단조롭게 느껴질 수 있습니다. 핵심 단어에 자연스러운 강세를 넣어보세요.",
    "해당 문장의 억양은 발표 흐름과 비슷하거나 자연스럽습니다.": "🟢 이 구간은 발표 흐름에 잘 어울리는 자연스러운 억양으로 전달되었습니다."
}

def analyze_intonation_from_audio(audio_path: str):
    stt_result = speech_recognizer(audio_path)
    segments = stt_result["segments"]
    text = stt_result["text"]
    intonation_results, avg_std, pitch_ranges = analyze_intonation(audio_path, segments)
    pitch_coverages = []
    pitch_stds = []
    feedback_types = []
    merged_results = []
    if segments:
        cur_start = round(segments[0]["start"], 2)
        cur_end = round(segments[0]["end"], 2)
        cur_text = segments[0].get("text") if segments[0].get("text") else "[문장 텍스트 없음]"
        cur_feedback = intonation_results[0]["intonation_feedback"]
        for i in range(1, len(segments)):
            seg = segments[i]
            feedback = intonation_results[i]["intonation_feedback"]
            seg_text = seg.get("text") if seg.get("text") else "[문장 텍스트 없음]"
            if feedback == cur_feedback:
                cur_end = round(seg["end"], 2)
                cur_text = cur_text.rstrip() + " " + seg_text.lstrip()
            else:
                merged_results.append({
                    "start": cur_start,
                    "end": cur_end,
                    "text": cur_text.strip(),
                    "feedback": FEEDBACK_MAP.get(cur_feedback, cur_feedback)
                })
                cur_start = round(seg["start"], 2)
                cur_end = round(seg["end"], 2)
                cur_text = seg_text
                cur_feedback = feedback
        merged_results.append({
            "start": cur_start,
            "end": cur_end,
            "text": cur_text.strip(),
            "feedback": FEEDBACK_MAP.get(cur_feedback, cur_feedback)
        })
    for i, res in enumerate(intonation_results):
        pitch_coverage = intonation_results[i]["pitch_coverage"]
        pitch_coverages.append(pitch_coverage)
        pitch_std = intonation_results[i]["pitch_std"]
        pitch_stds.append(pitch_std)
        feedback_types.append(intonation_results[i]["intonation_feedback"])
    avg_coverage = round(sum(pitch_coverages) / len(pitch_coverages), 2) if pitch_coverages else 0.0
    return {
        "full_text": text,
        "merged_feedback": merged_results,
        "avg_pitch_std": avg_std,
        "avg_pitch_coverage": avg_coverage
    }

# 터미널에 예쁘게 출력하는 함수

def print_intonation_result(result: dict):
    merged = result.get("merged_feedback", [])
    for idx, res in enumerate(merged):
        # 문장 번호 범위 계산
        start_idx = idx + 1
        end_idx = idx + 1
        # 실제로는 병합된 문장 범위 추적이 필요하지만, 여기선 단일 문장 단위로 출력
        sent_num_str = f"문장 {start_idx}" if start_idx == end_idx else f"문장 {start_idx}~{end_idx}"
        print(f"📌 {sent_num_str} ({res['start']}s ~ {res['end']}s)")
        print(f"    🗣️ 텍스트: \"{res['text']}\"")
        print(f"    📝 피드백: {res['feedback']}")
        print("\n" + "-"*80 + "\n")
    print(f"📈 전체 평균 pitch 검출률: {result.get('avg_pitch_coverage', 0)}%")
    print(f"🔍 전체 pitch 표준편차 평균: {result.get('avg_pitch_std', 0)}\n")

# 사람이 읽기 좋은 포맷의 문자열을 반환하는 함수

def get_pretty_intonation_result(result: dict) -> str:
    lines = []
    merged = result.get("merged_feedback", [])
    for idx, res in enumerate(merged):
        sent_num_str = f"문장 {idx+1}"
        lines.append(f"📌 {sent_num_str} ({res['start']}s ~ {res['end']}s)")
        lines.append(f"    🗣️ 텍스트: \"{res['text']}\"")
        lines.append(f"    📝 피드백: {res['feedback']}")
        lines.append("")
    lines.append(f"📈 전체 평균 pitch 검출률: {result.get('avg_pitch_coverage', 0)}%")
    lines.append(f"🔍 전체 pitch 표준편차 평균: {result.get('avg_pitch_std', 0)}")
    return "\n".join(lines) 