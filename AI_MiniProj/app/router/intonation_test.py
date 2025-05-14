import sys
from stt_service import speech_recognizer
from intonation_analyzer import analyze_intonation
import matplotlib.pyplot as plt
import platform

# 피드백 문구 변환 딕셔너리
FEEDBACK_MAP = {
    "이 문장은 억양 변화 폭이 지나치게 커서 과장된 느낌을 줄 수 있습니다.": "❗ 이 구간은 억양 변화 폭이 다소 커서 과장되게 들릴 수 있습니다. 중요한 내용 외에는 억양을 더 부드럽게 조절해보세요.",
    "이 문장은 발표 전체에 비해 억양이 특히 단조롭게 들릴 수 있습니다.": "❌ 이 구간은 억양 변화가 적어 단조롭게 느껴질 수 있습니다. 핵심 단어에 자연스러운 강세를 넣어보세요.",
    "해당 문장의 억양은 발표 흐름과 비슷하거나 자연스럽습니다.": "🟢 이 구간은 발표 흐름에 잘 어울리는 자연스러운 억양으로 전달되었습니다."
}

# 한글 폰트 설정 (Malgun Gothic, AppleGothic)
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python test_intonation.py <wav_파일_경로>")
        sys.exit(1)
    wav_path = sys.argv[1]

    # 1. STT 수행
    stt_result = speech_recognizer(wav_path)
    segments = stt_result["segments"]
    text = stt_result["text"]

    print("[전체 텍스트]")
    print(text)
    print()

    # 2. 억양 분석
    intonation_results, avg_std, pitch_ranges = analyze_intonation(wav_path, segments)

    print("[문장별 정보]")
    pitch_coverages = []
    pitch_stds = []
    feedback_types = []
    # 병합 로직
    merged_results = []
    if segments:
        cur_start = round(segments[0]["start"], 2)
        cur_end = round(segments[0]["end"], 2)
        cur_text = segments[0].get("text") if segments[0].get("text") else "[문장 텍스트 없음]"
        cur_feedback = intonation_results[0]["intonation_feedback"]
        for i in range(1, len(segments)):
            seg = segments[i]
            feedback = intonation_results[i]["intonation_feedback"]
            text = seg.get("text") if seg.get("text") else "[문장 텍스트 없음]"
            if feedback == cur_feedback:
                cur_end = round(seg["end"], 2)
                cur_text += text
            else:
                merged_results.append({
                    "start": cur_start,
                    "end": cur_end,
                    "text": cur_text,
                    "feedback": cur_feedback
                })
                cur_start = round(seg["start"], 2)
                cur_end = round(seg["end"], 2)
                cur_text = text
                cur_feedback = feedback
        # 마지막 병합 결과 추가
        merged_results.append({
            "start": cur_start,
            "end": cur_end,
            "text": cur_text,
            "feedback": cur_feedback
        })
    # 병합된 결과 출력 (피드백 문구 변환)
    for res in merged_results:
        feedback_msg = FEEDBACK_MAP.get(res['feedback'], res['feedback'])
        print(f"시작: {res['start']}s, 끝: {res['end']}s, 문장: \"{res['text']}\", 피드백: {feedback_msg}")

    # 전체 평균 pitch 검출률 출력
    for i, res in enumerate(intonation_results):
        pitch_coverage = intonation_results[i]["pitch_coverage"]
        pitch_coverages.append(pitch_coverage)
        pitch_std = intonation_results[i]["pitch_std"]
        pitch_stds.append(pitch_std)
        feedback_types.append(intonation_results[i]["intonation_feedback"])
    if pitch_coverages:
        avg_coverage = round(sum(pitch_coverages) / len(pitch_coverages), 2)
        print(f"\n전체 평균 pitch 검출률: {avg_coverage}%")
    # 전체 평균 pitch 표준편차 출력
    print(f"🔍 발표 전체 평균 pitch 표준편차: {avg_std}")

    # ===== 한글 친화적 시각화 코드 (막대그래프) =====
    x_labels = [f"문장 {i+1}" for i in range(len(pitch_stds))]
    bar_colors = []
    for fb in feedback_types:
        if "과장" in fb or "커서 과장" in fb or "폭이 다소 커서" in fb:
            bar_colors.append('red')
        elif "단조" in fb:
            bar_colors.append('blue')
        elif "자연" in fb or "잘 어울리는" in fb:
            bar_colors.append('green')
        else:
            bar_colors.append('gray')
    plt.figure(figsize=(14, 6))
    plt.bar(x_labels, pitch_stds, color=bar_colors)
    plt.axhline(avg_std, color='orange', linestyle='--', label='평균 표준편차')
    plt.title('문장별 억양 분석 결과')
    plt.xlabel('문장 번호')
    plt.ylabel('pitch 표준편차')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show() 