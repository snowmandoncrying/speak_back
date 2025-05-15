import numpy as np
import re

def count_syllables_korean(text):
    # 한글 음절(가-힣) 개수 세기
    return len(re.findall(r'[가-힣]', text))

def count_words_korean(text):
    # 띄어쓰기 기준 단어 수
    return len(text.strip().split())

def analyze_speed(wav_path, segments):
    results = []
    speeds = []
    word_speeds = []
    total_words = 0
    total_duration = 0
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        if duration < 1.0:
            continue  # 1초 미만 segment는 분석 제외
        text = seg.get("text") if seg.get("text") else ""
        syllable_count = count_syllables_korean(text)
        word_count = count_words_korean(text)
        speed_syllable = round(syllable_count / duration, 2) if duration > 0 else 0.0
        speed_word = round(word_count / duration, 2) if duration > 0 else 0.0
        speeds.append(speed_syllable)
        word_speeds.append(speed_word)
        total_words += word_count
        total_duration += duration
        # 피드백 분기
        if speed_syllable < 3:
            feedback = "❗ 발화 속도가 느린 편입니다. 조금 더 또박또박, 리듬감 있게 말해보세요."
        elif speed_syllable > 7:
            feedback = "❗ 발화 속도가 빠른 편입니다. 중요한 부분은 천천히 또박또박 말해보세요."
        else:
            feedback = "🟢 적절한 발화 속도로 전달되고 있습니다."
        results.append({
            "speed_syllable": speed_syllable,
            "speed_word": speed_word,
            "duration": duration,
            "syllable_count": syllable_count,
            "word_count": word_count,
            "feedback": feedback
        })
    avg_speed = round(np.mean(speeds), 2) if speeds else 0.0
    wpm = round((total_words / total_duration) * 60, 2) if total_duration > 0 else 0.0
    return results, avg_speed, wpm 