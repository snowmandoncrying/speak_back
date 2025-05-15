# pip install praat-parselmouth
import librosa
import numpy as np
import parselmouth

def analyze_intonation(wav_path, segments):
    y, sr = librosa.load(wav_path, sr=None)
    results = []
    pitch_stds = []
    pitch_ranges = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        y_seg = y[start_sample:end_sample]
        duration = end - start
        if len(y_seg) > 0:
            # parselmouth로 pitch 추출
            snd = parselmouth.Sound(y_seg, sr)
            pitch_obj = snd.to_pitch()
            pitch_values = pitch_obj.selected_array['frequency']
            voiced_f0 = pitch_values[pitch_values > 0]
            if len(pitch_values) > 0:
                pitch_coverage = round((len(voiced_f0) / len(pitch_values)) * 100, 2)
            else:
                pitch_coverage = 0.0
            if len(voiced_f0) > 0:
                pitch_std = round(float(np.std(voiced_f0)), 2)
                pitch_range = round(float(np.max(voiced_f0) - np.min(voiced_f0)), 2)
                pitch_stds.append(pitch_std)
                pitch_ranges.append(pitch_range)
            else:
                pitch_std = None
                pitch_range = None
                pitch_ranges.append(None)
            results.append({"pitch_std": pitch_std, "pitch_range": pitch_range, "pitch_coverage": pitch_coverage, "duration": duration})
        else:
            pitch_std = None
            pitch_range = None
            pitch_coverage = 0.0
            pitch_ranges.append(None)
            results.append({"pitch_std": pitch_std, "pitch_range": pitch_range, "pitch_coverage": pitch_coverage, "duration": 0.0})
    # 전체 평균 pitch_std 계산
    valid_stds = [res["pitch_std"] for res in results if res["pitch_std"] is not None]
    avg_std = round(sum(valid_stds) / len(valid_stds), 2) if valid_stds else 0.0
    # 피드백 부여 (문장 길이, pitch 검출률, 평서문 고려)
    for idx, res in enumerate(results):
        seg = segments[idx]
        sentence_text = seg.get("text") if seg.get("text") else ""
        is_declarative = sentence_text.strip().endswith(("다", "니다", "."))
        if res["pitch_std"] is None or res["pitch_range"] is None:
            res["intonation_feedback"] = "음성에서 유의미한 피치(억양)를 감지하지 못했습니다."
        elif res["duration"] <= 1.5 or res["pitch_coverage"] < 60:
            res["intonation_feedback"] = "문장 길이나 pitch 검출률이 낮아 억양 평가를 생략합니다."
        elif res["pitch_range"] >= 150:
            if is_declarative:
                res["intonation_feedback"] = "평서문이지만 억양 변화 폭이 크나, 문장 맥락상 자연스러운 억양일 수 있습니다."
            else:
                res["intonation_feedback"] = "이 문장은 억양 변화 폭이 지나치게 커서 과장된 느낌을 줄 수 있습니다."
        elif res["pitch_std"] < avg_std - 10:
            res["intonation_feedback"] = "이 문장은 발표 전체에 비해 억양이 특히 단조롭게 들릴 수 있습니다."
        else:
            res["intonation_feedback"] = "해당 문장의 억양은 발표 흐름과 비슷하거나 자연스럽습니다."
    return results, avg_std, pitch_ranges 