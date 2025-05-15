from fastapi import UploadFile
from pydub import AudioSegment
import os
import tempfile


def convert_to_wav(file: UploadFile) -> str:
    """
    다양한 오디오 파일(mp3, m4a, ogg, webm 등)을 .wav로 변환하여 임시 파일 경로를 반환합니다.
    Args:
        file (UploadFile): 업로드된 오디오 파일
    Returns:
        str: 변환된 .wav 파일의 임시 경로
    """
    # 업로드 파일을 임시 파일로 저장
    suffix = os.path.splitext(file.filename)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
        temp_in.write(file.file.read())
        temp_in_path = temp_in.name

    # 변환될 .wav 임시 파일 경로 생성
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_out_path = temp_out.name
    temp_out.close()

    # 오디오 변환
    audio = AudioSegment.from_file(temp_in_path)
    audio.export(temp_out_path, format="wav")

    # 임시 입력 파일 삭제
    os.remove(temp_in_path)

    return temp_out_path 