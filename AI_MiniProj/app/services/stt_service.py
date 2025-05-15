import whisper

def speech_recognizer(wav_path):
    model = whisper.load_model("medium")  # 필요시 "small", "large" 등으로 변경 가능
    result = model.transcribe(wav_path, language="ko")
    full_text = " ".join([seg["text"] for seg in result.get("segments", [])])
    return {
        "text": full_text,
        "duration": result["segments"][-1]["end"] if result["segments"] else 0,
        "segments": result["segments"]
    } 