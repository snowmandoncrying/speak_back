import whisper

def run_whisper_transcribe(wav_path: str, model_size: str = "medium") -> dict:
    """
    Whisper로 음성 파일을 텍스트 및 segment 리스트로 변환합니다.
    Args:
        wav_path (str): .wav 파일 경로
        model_size (str): Whisper 모델 크기 (기본값: medium)
    Returns:
        dict: Whisper의 전체 결과 (result["text"], result["segments"] 등 포함)
    """
    model = whisper.load_model(model_size)
    # 더 강력한 필러/머뭇거림/감탄사 포함 프롬프트
    initial_prompt = (
        "이 음성은 한국어 발표입니다. "
        "음, 어, 아, 으음, 어어, 그니까, 아마, 그래서, 뭐냐면, 저기, 그게, 그러니까, 네, 예, 응, 혹시, 일단, "
        "그리고 모든 머뭇거림, 반복, 침묵, 감탄사, 말버릇 등도 절대 생략하지 말고, "
        "말한 그대로, 아주 짧은 소리까지 빠짐없이 전사하세요. "
        "특히 '음...', '어...', '아...' 같은 짧은 감탄사, 반복되는 단어, 중간에 끊기는 소리도 모두 포함하세요. "
        "자연스러운 문장으로 바꾸지 말고, 실제 들리는 대로 최대한 정확하게 적어주세요. "
        "(이 프롬프트는 필러, 머뭇거림, 반복, 침묵, 감탄사 등 모든 비유창성 요소를 빠짐없이 전사하도록 설계되었습니다.)"
    )
    result = model.transcribe(
        wav_path,
        language="ko",
        task="transcribe",
        verbose=False,
        temperature=0.0,
        beam_size=8,
        best_of=8,
        patience=1.5,
        length_penalty=0.8,
        no_speech_threshold=0.1,
        logprob_threshold=-3.0,
        condition_on_previous_text=False,
        suppress_tokens=[],
        word_timestamps=False,
        initial_prompt=initial_prompt
    )
    return result 