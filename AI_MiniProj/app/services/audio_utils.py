from fastapi import UploadFile
from pydub import AudioSegment
import os
import tempfile

def convert_to_wav(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
        temp_in.write(file.file.read())
        temp_in_path = temp_in.name

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_out_path = temp_out.name
    temp_out.close()

    # 🔧 오디오 변환 + 설정 보정
    audio = AudioSegment.from_file(temp_in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # ✔ Whisper용 포맷
    audio.export(temp_out_path, format="wav")

    os.remove(temp_in_path)
    return temp_out_path
