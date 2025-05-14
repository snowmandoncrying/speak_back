import sys
from stt_service import speech_recognizer
from speed_analyzer import analyze_speed

def group_sentences_by_feedback(segments, speed_results):
    grouped_sentences = []
    current_group = []
    result_idx = 0
    
    for i, seg in enumerate(segments):
        start = round(seg["start"], 2)
        end = round(seg["end"], 2)
        sentence_text = seg.get("text") if seg.get("text") else "[문장 텍스트 없음]"
        duration = end - start
        
        if duration < 1.0:
            continue
            
        speed = speed_results[result_idx]["speed_syllable"]
        feedback = speed_results[result_idx]["feedback"]
        
        if not current_group or current_group[-1]["feedback"] == feedback:
            current_group.append({
                "index": i + 1,
                "start": start,
                "end": end,
                "text": sentence_text,
                "feedback": feedback
            })
        else:
            if current_group:
                grouped_sentences.append(current_group)
            current_group = [{
                "index": i + 1,
                "start": start,
                "end": end,
                "text": sentence_text,
                "feedback": feedback
            }]
        result_idx += 1
    
    if current_group:
        grouped_sentences.append(current_group)
    
    return grouped_sentences

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python speed_test.py <wav_파일_경로>")
        sys.exit(1)
    wav_path = sys.argv[1]

    # 1. STT 수행
    stt_result = speech_recognizer(wav_path)
    segments = stt_result["segments"]
    text = stt_result["text"]

    print("[전체 텍스트]")
    print(text)
    print()

    # 2. 발화 속도 분석
    speed_results, avg_speed, wpm = analyze_speed(wav_path, segments)

    # 3. 피드백 기반 문장 그룹화
    grouped_sentences = group_sentences_by_feedback(segments, speed_results)

    print("[그룹화된 발화 속도 정보]")
    for group in grouped_sentences:
        sentence_indices = ", ".join([f"문장 {s['index']}" for s in group])
        start_time = group[0]["start"]
        end_time = group[-1]["end"]
        combined_text = "".join([s["text"] for s in group])
        feedback = group[0]["feedback"]
        
        print(f"{sentence_indices}: 시작={start_time}s, 끝={end_time}s, 문장=\"{combined_text}\", 피드백={feedback}")
    
    print(f"\n전체 평균 초당 음절수: {avg_speed}")
    print(f"전체 WPM(단어/분): {wpm}") 