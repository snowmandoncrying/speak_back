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
    # FillerDetector에서 사용한 advanced 옵션과 initial_prompt 적용
    initial_prompt = (
        "한국어로 말하는 음성입니다. "
        "음, 어, 아, 으음, 어어, 그니까, 아마, 그래서, 뭐냐면, 저기, 그게, 그러니까, 네, 예, 응, 혹시, 일단 "
        "등의 모든 말버릇과 감탄사를 절대 생략하지 말고 말한 그대로 정확히 전사하세요. "
        "아주 짧은 말버릇이나 머뭇거림도 모두 포함해주세요. "
        "심지어 '음...', '어...', '아...' 같은 짧은 감탄사도 빠짐없이 전사하세요."
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