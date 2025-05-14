import librosa
import numpy as np
import parselmouth
from faster_whisper import WhisperModel
import time
import tempfile
import soundfile as sf
import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.spatial.distance import cosine
import warnings
import hashlib
import torch.nn.functional as F
import re
from pydub import AudioSegment
import edge_tts
import asyncio
import subprocess

warnings.filterwarnings('ignore')

class AdvancedSpeechAnalyzer:
    def __init__(self):
        print("🔄 시스템 초기화 중...")
        # Whisper 모델 초기화 (medium 모델 사용)
        self.model = WhisperModel("medium", device="cpu", compute_type="int8")
        # Wav2Vec2 모델 초기화
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec_model.eval()
        print("✅ 초기화 완료!")

    def validate_text(self, text: str) -> bool:
        """기준 텍스트 유효성 검사"""
        if not text or not text.strip():
            print("❌ 기준 텍스트가 비어있습니다.")
            return False
        return True

    def validate_audio_file(self, file_path: str) -> bool:
        """오디오 파일 유효성 검사"""
        try:
            if not os.path.exists(file_path):
                return False
            
            y, sr = librosa.load(file_path, sr=None, duration=0.1)
            return len(y) > 0 and sr > 0
        
        except Exception:
            return False

    def split_into_sentences(self, text: str) -> list:
        """텍스트를 문장 단위로 분리"""
        # 1. 텍스트 정규화
        text = text.strip()
        if not text:
            return []
        
        # 2. 문장 분리 (마침표, 물음표, 느낌표 기준)
        sentences = re.split(r'[.!?]+', text)
        # 빈 문장 제거 및 공백 정리
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def convert_to_wav(self, input_path: str, output_path: str = None) -> str:
        """MP3/WAV 파일을 올바른 WAV 형식으로 변환"""
        try:
            if output_path is None:
                output_path = os.path.splitext(input_path)[0] + "_converted.wav"
            
            # ffmpeg를 사용하여 16kHz, mono, PCM WAV로 변환
            command = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                output_path
            ]
            
            # ffmpeg 실행 (출력 숨김)
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ 파일 변환 실패: {result.stderr}")
                return None
            
            print(f"🔄 파일 변환 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 파일 변환 중 오류 발생: {e}")
            return None

    async def create_reference_audio(self, text: str, output_path: str = "reference.wav") -> tuple:
        """전체 기준 음성 생성 (edge-tts 사용)"""
        try:
            print("\n🎯 기준 음성 생성 중...")
            
            # 1. 텍스트 문장 분리
            sentences = self.split_into_sentences(text)
            if not sentences:
                print("❌ 유효한 문장이 없습니다.")
                return None, 0.0
            
            print(f"✅ 총 {len(sentences)}개 문장 처리 예정")
            
            # 2. 문장별 음성 생성 및 결합
            combined_audio = None
            temp_files = []
            
            for i, sentence in enumerate(sentences, 1):
                print(f"\n🔄 문장 {i}/{len(sentences)} 처리 중...")
                print(f"📝 텍스트: {sentence}")
                
                # 2.1 임시 파일 경로 생성 (MP3)
                temp_mp3 = os.path.join(tempfile.gettempdir(), f"temp_{i}.mp3")
                temp_wav = os.path.join(tempfile.gettempdir(), f"temp_{i}.wav")
                temp_files.extend([temp_mp3, temp_wav])
                
                try:
                    # 2.2 edge-tts로 음성 생성 (MP3)
                    communicate = edge_tts.Communicate(sentence, "ko-KR-SunHiNeural")
                    await communicate.save(temp_mp3)
                    print(f"✅ TTS 생성 (MP3): {temp_mp3}")
                    
                    # 2.3 MP3를 WAV로 변환
                    converted_wav = self.convert_to_wav(temp_mp3, temp_wav)
                    if not converted_wav:
                        continue
                    
                    # 2.4 음성 결합
                    segment = AudioSegment.from_wav(converted_wav)
                    if combined_audio is None:
                        combined_audio = segment
                    else:
                        # 문장 사이 0.5초 간격 추가
                        combined_audio = combined_audio + AudioSegment.silent(duration=500) + segment
                        
                    print("✅ 음성 생성 및 변환 성공")
                    
                except Exception as e:
                    print(f"❌ 문장 처리 실패: {e}")
                    continue
            
            # 3. 결과 저장
            if combined_audio is None:
                print("❌ 모든 문장 처리가 실패했습니다.")
                return None, 0.0
            
            try:
                # 3.1 결합된 음성을 MP3로 저장
                temp_combined_mp3 = os.path.join(tempfile.gettempdir(), "combined.mp3")
                combined_audio.export(temp_combined_mp3, format="mp3")
                
                # 3.2 최종 WAV 변환
                final_wav = self.convert_to_wav(temp_combined_mp3, output_path)
                if not final_wav:
                    return None, 0.0
                
                # 3.3 길이 계산
                duration = len(combined_audio) / 1000.0  # ms → 초 변환
                
                print(f"\n✅ 기준 음성 생성 완료: {output_path}")
                print(f"⏱️ 총 길이: {duration:.1f}초")
                print(f"🎧 기준 발음을 들으려면 해당 파일을 확인하세요.")
                
                return output_path, duration
                
            except Exception as e:
                print(f"❌ 최종 파일 저장 실패: {e}")
                return None, 0.0
                
            finally:
                # 4. 임시 파일 정리
                for temp_file in temp_files + [temp_combined_mp3]:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except:
                        pass
                    
        except Exception as e:
            print(f"❌ 기준 음성 생성 중 오류 발생: {e}")
            return None, 0.0

    def load_audio_for_wav2vec(self, audio_path: str, start_time: float = None, end_time: float = None) -> torch.Tensor:
        """오디오 파일을 로드하고 Wav2Vec2 입력 형식으로 변환"""
        try:
            # 1. 파일 존재 확인
            if not os.path.exists(audio_path):
                print(f"❌ 파일을 찾을 수 없음: {audio_path}")
                return None
                
            # 2. 파일 형식 확인
            file_ext = os.path.splitext(audio_path)[1].lower()
            supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
            if file_ext not in supported_formats:
                print(f"❌ 지원하지 않는 파일 형식: {file_ext}")
                return None
                
            # 3. 오디오 로드
            try:
                y, sr = librosa.load(audio_path, sr=16000)
            except Exception as e:
                print(f"❌ 오디오 로드 실패: {e}")
                return None
                
            # 4. 구간 추출 (지정된 경우)
            if start_time is not None and end_time is not None:
                if start_time >= len(y)/sr or end_time > len(y)/sr:
                    print(f"❌ 잘못된 구간 범위: {start_time}~{end_time}초 (전체 길이: {len(y)/sr:.1f}초)")
                    return None
                    
                start_idx = int(start_time * sr)
                end_idx = int(end_time * sr)
                y = y[start_idx:end_idx]
                
                if len(y) == 0:
                    print("❌ 추출된 구간이 비어있음")
                    return None
                
            # 5. 정규화 및 텐서 변환
            if len(y) == 0:
                print("❌ 오디오 데이터가 비어있음")
                return None
            
            y = librosa.util.normalize(y)
            audio_tensor = torch.FloatTensor(y)
            
            # 6. 채널 차원 추가
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            return audio_tensor
            
        except Exception as e:
            print(f"❌ 오디오 처리 중 오류 발생: {e}")
            return None

    def extract_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """음성 임베딩 추출 (Wav2Vec2 사용)"""
        try:
            if waveform is None or waveform.numel() == 0:
                print("❌ 입력 데이터가 비어있음")
                return None
            
            # 1. 모델 로드 (싱글톤 패턴)
            if not hasattr(self, 'wav2vec2_model'):
                try:
                    self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                    self.wav2vec2_model.eval()
                    if torch.cuda.is_available():
                        self.wav2vec2_model = self.wav2vec2_model.cuda()
                except Exception as e:
                    print(f"❌ Wav2Vec2 모델 로드 실패: {e}")
                    return None
                
            # 2. GPU 사용 가능 시 데이터 이동
            if torch.cuda.is_available():
                waveform = waveform.cuda()
            
            # 3. 메모리 관리를 위한 배치 처리
            with torch.no_grad():
                try:
                    # 배치 크기 계산 (메모리 한계 고려)
                    max_length = 30 * 16000  # 30초
                    if waveform.shape[1] > max_length:
                        segments = []
                        for i in range(0, waveform.shape[1], max_length):
                            segment = waveform[:, i:i+max_length]
                            features = self.wav2vec2_model(segment).last_hidden_state  # [1, seq_len, hidden_dim]
                            # 시퀀스 차원에 대해 평균 풀링
                            pooled = features.squeeze(0).mean(dim=0)  # [hidden_dim]
                            segments.append(pooled)
                        # 모든 세그먼트의 평균
                        embeddings = torch.stack(segments).mean(dim=0)
                    else:
                        features = self.wav2vec2_model(waveform).last_hidden_state  # [1, seq_len, hidden_dim]
                        # 시퀀스 차원에 대해 평균 풀링
                        embeddings = features.squeeze(0).mean(dim=0)  # [hidden_dim]
                    
                    # 4. 임베딩 검증
                    if embeddings.dim() != 1:
                        print(f"❌ 잘못된 임베딩 차원: {embeddings.shape}")
                        return None
                        
                    if embeddings.shape[0] != 768:  # Wav2Vec2-base의 hidden_dim
                        print(f"❌ 잘못된 임베딩 크기: {embeddings.shape}")
                        return None
                        
                    # 5. L2 정규화
                    embeddings = F.normalize(embeddings, p=2, dim=0)
                    return embeddings.cpu()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("❌ GPU 메모리 부족, CPU로 전환")
                        torch.cuda.empty_cache()
                        self.wav2vec2_model = self.wav2vec2_model.cpu()
                        waveform = waveform.cpu()
                        features = self.wav2vec2_model(waveform).last_hidden_state
                        embeddings = features.squeeze(0).mean(dim=0)
                        embeddings = F.normalize(embeddings, p=2, dim=0)
                        return embeddings
                    raise
                
        except Exception as e:
            print(f"❌ 임베딩 추출 중 오류 발생: {e}")
            return None

    def calculate_similarity(self, user_emb: torch.Tensor, ref_emb: torch.Tensor) -> float:
        """코사인 유사도 계산 및 점수화"""
        try:
            if user_emb is None or ref_emb is None:
                return 0
            
            # 1. 임베딩 검증
            if user_emb.dim() != 1 or ref_emb.dim() != 1:
                print(f"⚠️ 잘못된 임베딩 차원: 사용자({user_emb.dim()}) vs 기준({ref_emb.dim()})")
                return 0
            
            if user_emb.shape != ref_emb.shape:
                print(f"⚠️ 임베딩 크기 불일치: 사용자({user_emb.shape}) vs 기준({ref_emb.shape})")
                return 0
            
            # 2. 코사인 유사도 계산
            similarity = F.cosine_similarity(user_emb.unsqueeze(0), ref_emb.unsqueeze(0), dim=1).item()
            
            # 3. 점수 변환 (0~100)
            score = max(0, min(100, (similarity + 1) * 50))  # [-1, 1] → [0, 100]
            return score
            
        except Exception as e:
            print(f"❌ 유사도 계산 오류: {e}")
            return 0

    def get_similarity_feedback(self, score: float) -> str:
        """점수에 따른 피드백 생성"""
        if score >= 85:
            return "발음이 매우 정확하고 자연스럽습니다."
        elif score >= 65:
            return "발음이 대체로 정확하고 이해하기 쉽습니다."
        elif score >= 40:
            return "일부 발음이 불명확하나 의미 전달은 가능합니다."
        else:
            return "발음이 불명확하여 개선이 필요합니다."

    def analyze_pronunciation(self, audio_path: str, start_time: float, end_time: float) -> dict:
        """발음 품질 분석 (Parselmouth - fallback용)"""
        try:
            # 1. 오디오 로드 및 구간 추출
            try:
                y, sr = librosa.load(audio_path)
                duration = len(y) / sr
            except Exception as e:
                return {
                    "quality": "분석 불가",
                    "feedback": f"오디오 파일을 로드할 수 없습니다: {str(e)}",
                    "score": 0.0
                }
            
            if start_time >= duration or end_time > duration:
                return {
                    "quality": "분석 불가",
                    "feedback": f"요청한 구간이 오디오 길이를 초과합니다: {start_time:.1f}~{end_time:.1f}초 (전체 길이: {duration:.1f}초)",
                    "score": 0.0
                }
            
            start_sample = max(0, int(start_time * sr))
            end_sample = min(len(y), int(end_time * sr))
            segment = y[start_sample:end_sample]
            
            if len(segment) == 0:
                return {
                    "quality": "분석 불가",
                    "feedback": "추출된 구간이 비어있습니다.",
                    "score": 0.0
                }
            
            # 2. 임시 WAV 파일 처리
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                try:
                    sf.write(temp_wav.name, segment, sr)
                    sound = parselmouth.Sound(temp_wav.name)
                except Exception as e:
                    return {
                        "quality": "분석 불가",
                        "feedback": f"음성 파일 변환 중 오류가 발생했습니다: {str(e)}",
                        "score": 0.0
                    }
                    
                try:
                    # 3. 피치 분석
                    pitch = sound.to_pitch()
                    pitch_values = pitch.selected_array['frequency']
                    valid_pitch = pitch_values[pitch_values > 0]
                    
                    if len(valid_pitch) == 0:
                        return {
                            "quality": "분석 불가",
                            "feedback": "발음을 감지할 수 없습니다.",
                            "score": 0.0
                        }
                    
                    # 4. 음성 특성 분석
                    pitch_std = np.std(valid_pitch)
                    
                    formant = sound.to_formant_burg()
                    f1_values = [formant.get_value_at_time(1, t) for t in formant.xs()]
                    f2_values = [formant.get_value_at_time(2, t) for t in formant.xs()]
                    
                    f1_std = np.std([f for f in f1_values if f != 0])
                    f2_std = np.std([f for f in f2_values if f != 0])
                    
                    # 5. 점수 계산
                    stability_score = 100 - (pitch_std * 0.05 + f1_std * 0.025 + f2_std * 0.025)
                    stability_score = max(0, min(100, stability_score))
                    
                    # 6. 품질 평가
                    if stability_score >= 80:
                        quality = "우수"
                        feedback = "발음이 매우 명확하고 안정적입니다."
                    elif stability_score >= 65:
                        quality = "양호"
                        feedback = "발음이 대체로 명확하고 이해하기 쉽습니다."
                    elif stability_score >= 50:
                        quality = "보통"
                        feedback = "발음이 이해할 만하나, 일부 개선의 여지가 있습니다."
                    else:
                        quality = "미흡"
                        feedback = "발음이 불안정하여 개선이 필요합니다."
                    
                    return {
                        "quality": quality,
                        "feedback": feedback,
                        "score": stability_score
                    }
                    
                except Exception as e:
                    return {
                        "quality": "분석 불가",
                        "feedback": f"음성 특성 분석 중 오류가 발생했습니다: {str(e)}",
                        "score": 0.0
                    }
                finally:
                    try:
                        os.remove(temp_wav.name)
                    except:
                        pass
                    
        except Exception as e:
            return {
                "quality": "분석 실패",
                "feedback": f"발음 분석 중 오류가 발생했습니다: {str(e)}",
                "score": 0.0
            }

    def analyze_speech_rate(self, segment) -> tuple:
        """말속도(WPM) 분석"""
        try:
            # 단어 수 계산 (공백 기준)
            words = len(segment.text.split())
            duration = segment.end - segment.start
            
            if duration <= 0 or words == 0:
                return None, None
            
            # WPM (Words Per Minute) 계산
            wpm = (words / duration) * 60
            
            # 속도 판정
            if wpm < 90:
                feedback = "말속도가 다소 느립니다. 조금 더 자연스러운 속도로 말해보세요."
                speed = "느림"
            elif wpm <= 140:
                feedback = "적절한 말속도로 청중이 이해하기 좋습니다."
                speed = "적절"
            else:
                feedback = "말속도가 다소 빠릅니다. 청중을 위해 조금 더 천천히 말해보세요."
                speed = "빠름"
            
            return wpm, (speed, feedback)
            
        except Exception as e:
            print(f"❌ 말속도 분석 오류: {e}")
            return None, None

    def detect_stuttering(self, text: str) -> tuple:
        """말더듬기 감지"""
        try:
            import re
            
            # 연속된 단어/음절 반복 패턴 찾기
            # 1. "아, 아, 아" 형태
            pattern1 = r'(\w+)[,\s]+\1[,\s]+\1+'
            # 2. "저기.. 저기" 형태
            pattern2 = r'(\w+)[.]{2,}\s+\1+'
            
            matches = []
            for pattern in [pattern1, pattern2]:
                found = re.finditer(pattern, text)
                for match in found:
                    matches.append(match.group(1))
                
            if matches:
                repeated_words = ', '.join(set(matches))
                feedback = f"'{repeated_words}' 부분에서 말더듬이 감지됩니다. 긴장을 풀고 천천히 말해보세요."
                return repeated_words, feedback
            
            return None, None
            
        except Exception as e:
            print(f"❌ 말더듬기 감지 오류: {e}")
            return None, None

    def analyze_silence(self, current_segment, previous_segment) -> tuple:
        """문장 간 침묵 감지"""
        try:
            if not previous_segment:
                return None, None
            
            silence_duration = current_segment.start - previous_segment.end
            
            # 0.5초 이상의 침묵을 감지
            if silence_duration >= 0.5:
                feedback = "불필요한 침묵이 발생했습니다. 더 자연스럽게 이어 말해보세요."
                return silence_duration, feedback
            
            return None, None
            
        except Exception as e:
            print(f"❌ 침묵 감지 오류: {e}")
            return None, None

    def compare_segment_with_reference(self, user_audio: str, ref_audio: str, segment, ref_duration: float, previous_segment=None) -> dict:
        """문장 구간별 비교 분석"""
        try:
            # 1. 구간 시간 계산 (padding 0.2초 추가)
            start_time = max(0, segment.start - 0.2)
            end_time = min(segment.end + 0.2, ref_duration)
            
            # 구간 유효성 검사
            if start_time >= end_time:
                return {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "similarity_score": 0.0,
                    "similarity_feedback": "비교 불가: 잘못된 구간 범위",
                    "pronunciation_quality": "분석 불가",
                    "pronunciation_score": 0.0,
                    "pronunciation_feedback": f"잘못된 구간 범위: {start_time:.1f}~{end_time:.1f}초"
                }
            
            if end_time > ref_duration:
                return {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "similarity_score": 0.0,
                    "similarity_feedback": "비교 불가: 기준 음성 길이 초과",
                    "pronunciation_quality": "분석 불가",
                    "pronunciation_score": 0.0,
                    "pronunciation_feedback": f"기준 음성 길이({ref_duration:.1f}초)를 초과하는 구간입니다."
                }
            
            print(f"\n🟩 문장: {segment.text}")
            print(f"⏱️ 구간: {start_time:.1f}초 ~ {end_time:.1f}초")
            
            # 2. 사용자 음성 구간 로드
            user_wave = self.load_audio_for_wav2vec(user_audio, start_time, end_time)
            if user_wave is None:
                return {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "similarity_score": 0.0,
                    "similarity_feedback": "비교 불가: 사용자 음성 로드 실패",
                    "pronunciation_quality": "분석 불가",
                    "pronunciation_score": 0.0,
                    "pronunciation_feedback": "사용자 음성 구간을 로드할 수 없습니다."
                }
            print("✅ 사용자 구간 추출 성공")
            
            # 3. 기준 음성 구간 로드
            ref_wave = self.load_audio_for_wav2vec(ref_audio, start_time, end_time)
            if ref_wave is None:
                return {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "similarity_score": 0.0,
                    "similarity_feedback": "비교 불가: 기준 음성 로드 실패",
                    "pronunciation_quality": "분석 불가",
                    "pronunciation_score": 0.0,
                    "pronunciation_feedback": "기준 음성 구간을 로드할 수 없습니다."
                }
            print("✅ AI 구간 추출 성공")
            
            # 4. 임베딩 추출
            user_emb = self.extract_embeddings(user_wave)
            if user_emb is None:
                return {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "similarity_score": 0.0,
                    "similarity_feedback": "비교 불가: 사용자 음성 분석 실패",
                    "pronunciation_quality": "분석 불가",
                    "pronunciation_score": 0.0,
                    "pronunciation_feedback": "사용자 음성의 특성을 추출할 수 없습니다."
                }
            
            ref_emb = self.extract_embeddings(ref_wave)
            if ref_emb is None:
                return {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "similarity_score": 0.0,
                    "similarity_feedback": "비교 불가: 기준 음성 분석 실패",
                    "pronunciation_quality": "분석 불가",
                    "pronunciation_score": 0.0,
                    "pronunciation_feedback": "기준 음성의 특성을 추출할 수 없습니다."
                }
            
            # 5. 유사도 계산
            score = self.calculate_similarity(user_emb, ref_emb)
            feedback = self.get_similarity_feedback(score)
            print(f"📊 유사도: {score:.1f} / 100")
            
            # 6. Parselmouth 분석 추가 (보조 지표)
            parselmouth_result = self.analyze_pronunciation(user_audio, start_time, end_time)
            
            # 7. 음량 분석 추가
            volume_result = self.analyze_volume(user_audio, start_time, end_time)
            
            # 기본 분석 결과
            result = {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "similarity_score": score,
                "similarity_feedback": feedback,
                "pronunciation_quality": parselmouth_result["quality"],
                "pronunciation_score": parselmouth_result["score"],
                "pronunciation_feedback": parselmouth_result["feedback"]
            }
            
            # 1. 말속도 분석
            wpm, speed_info = self.analyze_speech_rate(segment)
            if wpm is not None:
                result["speech_rate"] = {
                    "wpm": wpm,
                    "status": speed_info[0],
                    "feedback": speed_info[1]
                }
            
            # 2. 말더듬기 감지
            stutter_words, stutter_feedback = self.detect_stuttering(segment.text)
            if stutter_words is not None:
                result["stuttering"] = {
                    "repeated_words": stutter_words,
                    "feedback": stutter_feedback
                }
            
            # 3. 침묵 감지
            silence_duration, silence_feedback = self.analyze_silence(segment, previous_segment)
            if silence_duration is not None:
                result["silence"] = {
                    "duration": silence_duration,
                    "feedback": silence_feedback
                }
            
            # 4. 음량 분석 결과 추가
            if volume_result is not None:
                result["volume"] = volume_result
                # 음량 분석 결과 출력
                print(f"\n🔊 음량 분석: 평균 RMS = {volume_result['rms']:.3f} → {volume_result['emoji']} {volume_result['status']}")
                print(f"📢 음량 피드백: {volume_result['feedback']}")
                
                # 음량 상태에 따른 추가 피드백
                if volume_result['status'] == "Too Quiet":
                    print("💡 개선 제안: ")
                    print("- 복식호흡을 활용하여 발성에 힘을 실어보세요")
                    print("- 청중을 향해 더 또렷하게 발성해보세요")
                elif volume_result['status'] == "Too Loud":
                    print("💡 개선 제안: ")
                    print("- 호흡을 안정시키고 차분하게 말해보세요")
                    print("- 마이크와의 거리를 약간 더 두어보세요")
                else:
                    print("💡 잘하고 있는 점: ")
                    print("- 청중이 듣기 편안한 음량을 잘 유지하고 있습니다")
                    print("- 이 음량을 계속 유지해주세요")
            
            return result
            
        except Exception as e:
            print(f"\n❌ 구간 비교 오류: {e}")
            # Fallback: Parselmouth 분석만 수행
            result = self.analyze_pronunciation(user_audio, segment.start, segment.end)
            return {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "similarity_score": 0.0,
                "similarity_feedback": f"비교 불가: {str(e)}",
                "pronunciation_quality": result["quality"],
                "pronunciation_score": result["score"],
                "pronunciation_feedback": result["feedback"]
            }

    def transcribe_audio(self, audio_path: str) -> list:
        """음성 파일을 문장 단위로 전사"""
        try:
            print("\n🎯 음성 전사 중...")
            
            # 1. 파일 유효성 검사
            if not self.validate_audio_file(audio_path):
                return []
            
            # 2. 전체 길이 확인
            duration = librosa.get_duration(filename=audio_path)
            print(f"✅ 전체 길이: {duration:.1f}초")
            
            # 3. 전사 수행
            segments, _ = self.model.transcribe(audio_path, language="ko")
            segments = list(segments)
            
            if not segments:
                print("⚠️ 감지된 문장이 없습니다.")
                return []
            
            print(f"✅ 전사 완료: {len(segments)}개 문장 감지")
            return segments
            
        except Exception as e:
            print(f"❌ 전사 오류: {e}")
            return []

    def generate_summary_statistics(self, results: list) -> dict:
        """분석 결과에 대한 요약 통계 생성"""
        stats = {
            "total_segments": len(results),
            "poor_pronunciation": {
                "count": 0,
                "segments": []  # (문장 번호, 시작 시간, 끝 시간) 튜플 리스트
            },
            "slow_speech": {
                "count": 0,
                "segments": []
            },
            "fast_speech": {
                "count": 0,
                "segments": []
            },
            "silence_detected": {
                "count": 0,
                "segments": []
            },
            "stuttering_detected": {
                "count": 0,
                "segments": []
            },
            "volume_too_quiet": {
                "count": 0,
                "segments": []
            },
            "volume_too_loud": {
                "count": 0,
                "segments": []
            },
            "volume_good": {
                "count": 0,
                "segments": []
            },
            "avg_similarity": 0.0,
            "avg_pronunciation": 0.0,
            "avg_speech_rate": 0.0,
            "avg_rms": 0.0,
            "valid_segments": 0
        }
        
        total_rms = 0.0
        rms_count = 0
        
        for i, result in enumerate(results, 1):
            # 기본 점수가 있는 경우만 카운트
            if result.get("pronunciation_score", 0) > 0:
                stats["valid_segments"] += 1
                
                # 발음 품질 분석
                if result["pronunciation_score"] < 65:
                    stats["poor_pronunciation"]["count"] += 1
                    stats["poor_pronunciation"]["segments"].append((i, result["start"], result["end"]))
                stats["avg_pronunciation"] += result["pronunciation_score"]
                
                # 유사도 분석
                if result.get("similarity_score", 0) > 0:
                    stats["avg_similarity"] += result["similarity_score"]
                
                # 말속도 분석
                if "speech_rate" in result:
                    wpm = result["speech_rate"]["wpm"]
                    stats["avg_speech_rate"] += wpm
                    if wpm < 90:
                        stats["slow_speech"]["count"] += 1
                        stats["slow_speech"]["segments"].append((i, result["start"], result["end"]))
                    elif wpm > 140:
                        stats["fast_speech"]["count"] += 1
                        stats["fast_speech"]["segments"].append((i, result["start"], result["end"]))
                
                # 침묵 감지
                if "silence" in result:
                    stats["silence_detected"]["count"] += 1
                    stats["silence_detected"]["segments"].append((i, result["start"], result["end"]))
                
                # 말더듬기 감지
                if "stuttering" in result:
                    stats["stuttering_detected"]["count"] += 1
                    stats["stuttering_detected"]["segments"].append((i, result["start"], result["end"]))
                
                # 음량 분석
                if "volume" in result:
                    volume = result["volume"]
                    total_rms += volume["rms"]
                    rms_count += 1
                    
                    if volume["status"] == "Too Quiet":
                        stats["volume_too_quiet"]["count"] += 1
                        stats["volume_too_quiet"]["segments"].append((i, result["start"], result["end"]))
                    elif volume["status"] == "Too Loud":
                        stats["volume_too_loud"]["count"] += 1
                        stats["volume_too_loud"]["segments"].append((i, result["start"], result["end"]))
                    else:  # 적절
                        stats["volume_good"]["count"] += 1
                        stats["volume_good"]["segments"].append((i, result["start"], result["end"]))
        
        # 평균값 계산
        if stats["valid_segments"] > 0:
            stats["avg_pronunciation"] /= stats["valid_segments"]
            stats["avg_speech_rate"] /= stats["valid_segments"]
            if stats.get("avg_similarity", 0) > 0:
                stats["avg_similarity"] /= stats["valid_segments"]
        
        # 평균 RMS 계산
        if rms_count > 0:
            stats["avg_rms"] = total_rms / rms_count
        
        return stats

    def generate_overall_feedback(self, stats: dict) -> str:
        """통계를 바탕으로 종합 피드백 생성"""
        feedback_parts = []
        
        # 1. 전반적인 발음 평가
        if stats["avg_pronunciation"] >= 80:
            feedback_parts.append("전반적으로 발음이 매우 안정적이고 명확합니다.")
        elif stats["avg_pronunciation"] >= 65:
            feedback_parts.append("전반적으로 발음이 안정적이며 이해하기 좋은 수준입니다.")
        else:
            feedback_parts.append("전반적으로 발음의 개선이 필요합니다.")
        
        # 2. 말속도 평가
        if stats["slow_speech"]["count"] > 0 or stats["fast_speech"]["count"] > 0:
            speed_issues = []
            if stats["slow_speech"]["count"] > 0:
                speed_issues.append(f"{stats['slow_speech']['count']}개 문장에서 말속도가 느리고")
            if stats["fast_speech"]["count"] > 0:
                speed_issues.append(f"{stats['fast_speech']['count']}개 문장에서 말속도가 빠르며")
            feedback_parts.append(f"총 {', '.join(speed_issues)},")
        else:
            feedback_parts.append("말속도가 전반적으로 적절하며,")
        
        # 3. 침묵과 말더듬 평가
        issues = []
        if stats["silence_detected"]["count"] > 0:
            issues.append(f"{stats['silence_detected']['count']}회의 불필요한 침묵")
        if stats["stuttering_detected"]["count"] > 0:
            issues.append(f"{stats['stuttering_detected']['count']}회의 말더듬")
        
        if issues:
            feedback_parts.append(f"{' 및 '.join(issues)}이 감지되어 전달력이 다소 저하될 수 있습니다.")
            feedback_parts.append("발표 흐름에 맞게 자연스러운 연결을 연습해보세요.")
        else:
            feedback_parts.append("전반적으로 자연스러운 발표 흐름을 보여줍니다.")
        
        # 4. 음량 평가 추가
        volume_feedback = []
        if stats["volume_too_quiet"]["count"] > 0:
            volume_feedback.append(f"{stats['volume_too_quiet']['count']}개 문장에서 목소리가 너무 작고")
        if stats["volume_too_loud"]["count"] > 0:
            volume_feedback.append(f"{stats['volume_too_loud']['count']}개 문장에서 목소리가 너무 큽니다")
        
        if volume_feedback:
            feedback_parts.append(f"\n음량 면에서는 {', '.join(volume_feedback)}.")
            if stats["volume_good"]["count"] > stats["total_segments"] * 0.7:  # 70% 이상이 적절한 경우
                feedback_parts.append("하지만 대부분의 문장에서 음량이 적절하여 전반적인 전달력은 좋습니다.")
            else:
                feedback_parts.append("전반적으로 음량 조절에 신경 쓰면 더 좋은 발표가 될 것 같습니다.")
        else:
            feedback_parts.append("\n음량이 전반적으로 매우 적절하여 청중이 듣기 편안한 발표입니다.")
        
        return " ".join(feedback_parts)

    def analyze_volume(self, audio_path: str, start_time: float, end_time: float) -> dict:
        """음량(RMS) 분석"""
        try:
            # 1. 오디오 로드 및 구간 추출
            y, sr = librosa.load(audio_path, sr=None)
            if start_time is not None and end_time is not None:
                start_idx = int(start_time * sr)
                end_idx = int(end_time * sr)
                y = y[start_idx:end_idx]

            # 2. RMS 에너지 계산 (프레임 단위)
            frame_length = 2048  # 약 0.1초 단위
            hop_length = 512     # 프레임 간 이동 간격
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 3. 유효한 음성 구간의 RMS만 선택 (매우 낮은 에너지는 제외)
            # 완전한 무음이 아닌, 매우 낮은 에너지 임계값 사용
            energy_threshold = 0.005  # 매우 낮은 임계값 설정
            valid_rms = rms[rms > energy_threshold]
            
            if len(valid_rms) == 0:  # 유효한 RMS가 없는 경우
                mean_rms = 0.0
            else:
                # 상위 80% 구간의 평균 RMS 사용 (너무 낮은 값 제외)
                mean_rms = np.percentile(valid_rms, 80)

            # 4. 음량 등급 분류 (기준값 미세 조정)
            if mean_rms < 0.015:
                volume_status = "Too Quiet"
                feedback = "목소리가 작습니다. 조금 더 힘 있게 말해보세요."
                emoji = "📉"
            elif mean_rms <= 0.05:
                volume_status = "적절"
                feedback = "목소리 크기가 적절하여 발표 전달력이 좋습니다."
                emoji = "✅"
            else:
                volume_status = "Too Loud"
                feedback = "목소리가 다소 커서 부담스러울 수 있습니다. 약간 낮춰보세요."
                emoji = "📈"

            return {
                "rms": mean_rms,
                "status": volume_status,
                "feedback": feedback,
                "emoji": emoji
            }

        except Exception as e:
            print(f"❌ 음량 분석 오류: {e}")
            return None

def get_audio_duration(audio_path: str) -> float:
    """오디오 파일의 전체 길이를 반환합니다."""
    try:
        if not os.path.exists(audio_path):
            print(f"❌ 파일이 존재하지 않습니다: {audio_path}")
            return 0.0
            
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration == 0.0:
            print(f"⚠️ 오디오 길이가 0초입니다: {audio_path}")
            
        return duration
        
    except Exception as e:
        print(f"❌ 오디오 길이 측정 오류: {e}")
        return 0.0

def get_quality_emoji(quality: str) -> str:
    """품질에 따른 이모지 반환"""
    quality_emojis = {
        "명확": "🌟",
        "양호": "✨",
        "보통": "⭐",
        "불명확": "💫",
        "분석 불가": "⚠️",
        "분석 실패": "❌"
    }
    return quality_emojis.get(quality, "✔️")

if __name__ == "__main__":
    import argparse
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="발음 평가 시스템")
    parser.add_argument("--audio", type=str, help="분석할 음성 파일 경로")
    parser.add_argument("--text", type=str, help="기준 텍스트")
    parser.add_argument("--text-file", type=str, help="기준 텍스트 파일 경로")
    parser.add_argument("--generate-only", action="store_true", help="기준 음성만 생성")
    args = parser.parse_args()
    
    try:
        # 기준 텍스트 로드
        reference_text = None
        if args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    reference_text = f.read().strip()
            except Exception as e:
                print(f"❌ 텍스트 파일 로드 실패: {e}")
                exit(1)
        elif args.text:
            reference_text = args.text.strip()
        else:
            print("\n📝 기준 텍스트를 입력하세요 (입력 완료 후 빈 줄에서 Ctrl+Z 또는 Ctrl+D):")
            try:
                lines = []
                while True:
                    try:
                        line = input()
                        lines.append(line)
                    except EOFError:
                        break
                reference_text = '\n'.join(lines).strip()
            except Exception as e:
                print(f"❌ 텍스트 입력 오류: {e}")
                exit(1)
                
        if not reference_text:
            print("❌ 기준 텍스트가 비어있습니다.")
            exit(1)

        print(f"\n📊 분석 시작")
        print(f"📝 기준 텍스트 길이: {len(reference_text)}자")

        # 분석기 초기화
        analyzer = AdvancedSpeechAnalyzer()
        start_process = time.time()
        
        # 기준 음성 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ref_path = f"reference_{timestamp}.wav"
        
        # edge-tts는 비동기 함수이므로 asyncio.run() 사용
        ref_path, ref_duration = asyncio.run(analyzer.create_reference_audio(reference_text, ref_path))
        if not ref_path:
            print("❌ 기준 음성 생성에 실패했습니다.")
            exit(1)
        
        print(f"\n💾 기준 음성이 저장되었습니다: {ref_path}")
        print("🎧 이 파일을 통해 AI가 생성한 표준 발음을 들어볼 수 있습니다.")

        # 기준 음성만 생성하는 모드인 경우 여기서 종료
        if args.generate_only:
            print("\n✅ 기준 음성 생성 완료!")
            exit(0)

        # 분석할 음성 파일 경로 확인
        if not args.audio:
            audio_path = input("\n분석할 음성 파일 경로를 입력하세요: ").strip()
        else:
            audio_path = args.audio
            
        if not os.path.exists(audio_path):
            print(f"❌ 음성 파일을 찾을 수 없습니다: {audio_path}")
            exit(1)
            
        print(f"🎤 사용자 음성: {audio_path}")

        # 음성 전사 및 분석
        segments = analyzer.transcribe_audio(audio_path)
        if not segments:
            print("❌ 음성 전사에 실패했습니다.")
            if os.path.exists(ref_path):
                os.remove(ref_path)
            exit(1)

        print("\n🎯 발음 평가 기준:")
        print("🌟 우수 (80점 이상): 발음이 매우 명확하고 안정적")
        print("✨ 양호 (65-79점): 발음이 대체로 명확하고 이해하기 쉬움")
        print("⭐ 보통 (50-64점): 발음이 이해할 만하나 개선의 여지 있음")
        print("💫 미흡 (50점 미만): 발음이 불안정하여 개선 필요")
        print("⚠️ 분석 불가: 구간 추출 또는 비교 실패")
        
        print("\n🔍 문장별 발음 분석 결과:")
        
        # 각 문장별 분석 수행
        total_segments = len(segments)
        successful_comparisons = 0
        total_similarity = 0.0
        total_pronunciation = 0.0
        previous_segment = None
        analysis_results = []  # 전체 분석 결과 저장
        
        for i, segment in enumerate(segments, 1):
            print(f"\n📝 문장 {i}/{total_segments} 분석 중...")
            
            result = analyzer.compare_segment_with_reference(audio_path, ref_path, segment, ref_duration, previous_segment)
            analysis_results.append(result)  # 결과 저장
            
            print(f"\n🟩 문장 {i}: {result['text']}")
            print(f"⏱️ 구간: {result['start']:.1f}초 ~ {result['end']:.1f}초")
            
            # AI 음성과의 유사도
            if result['similarity_score'] > 0:
                print(f"📊 AI 음성 유사도: {result['similarity_score']:.1f}/100")
                print(f"💡 유사도 피드백: {result['similarity_feedback']}")
                total_similarity += result['similarity_score']
                successful_comparisons += 1
            else:
                print("📊 AI 음성 비교: " + result['similarity_feedback'])
            
            # Parselmouth 분석 결과
            if result['pronunciation_score'] > 0:
                total_pronunciation += result['pronunciation_score']
                
            quality_emoji = get_quality_emoji(result['pronunciation_quality'])
            print(f"{quality_emoji} 발음 품질: {result['pronunciation_quality']} (점수: {result['pronunciation_score']:.1f}/100)")
            print(f"🗣️ 발음 피드백: {result['pronunciation_feedback']}")
            
            # 말속도 출력
            if 'speech_rate' in result:
                print(f"⏩ 말속도: {result['speech_rate']['wpm']:.1f} WPM → {result['speech_rate']['status']}")
                print(f"💬 속도 피드백: {result['speech_rate']['feedback']}")
            
            # 말더듬기 출력
            if 'stuttering' in result:
                print(f"🔁 말더듬기: 단어 '{result['stuttering']['repeated_words']}'가 반복됨")
                print(f"🧠 말더듬 피드백: {result['stuttering']['feedback']}")
            
            # 침묵 출력
            if 'silence' in result:
                print(f"🔇 침묵 감지: 이전 문장과의 간격 {result['silence']['duration']:.1f}초")
                print(f"🔎 침묵 피드백: {result['silence']['feedback']}")
            
            # 음량 출력
            if 'volume' in result:
                print(f"\n🔊 음량 분석: 평균 RMS = {result['volume']['rms']:.3f} → {result['volume']['emoji']} {result['volume']['status']}")
                print(f"📢 음량 피드백: {result['volume']['feedback']}")
                
                # 음량 상태에 따른 추가 피드백
                if result['volume']['status'] == "Too Quiet":
                    print("💡 개선 제안: ")
                    print("- 복식호흡을 활용하여 발성에 힘을 실어보세요")
                    print("- 청중을 향해 더 또렷하게 발성해보세요")
                elif result['volume']['status'] == "Too Loud":
                    print("💡 개선 제안: ")
                    print("- 호흡을 안정시키고 차분하게 말해보세요")
                    print("- 마이크와의 거리를 약간 더 두어보세요")
                else:
                    print("💡 잘하고 있는 점: ")
                    print("- 청중이 듣기 편안한 음량을 잘 유지하고 있습니다")
                    print("- 이 음량을 계속 유지해주세요")
            
            print("---------------")
            previous_segment = segment
        
        # 요약 통계 및 종합 피드백
        stats = analyzer.generate_summary_statistics(analysis_results)
        
        print("\n📊 실수 요약 통계:")
        
        # 발음 개선 필요 문장
        if stats["poor_pronunciation"]["count"] > 0:
            print(f"\n발음 개선 필요 문장 수: {stats['poor_pronunciation']['count']}개")
            segments_info = [f"문장 {seg[0]} ({seg[1]:.1f}~{seg[2]:.1f}초)" 
                           for seg in stats["poor_pronunciation"]["segments"]]
            print(f"→ {', '.join(segments_info)}")
        
        # 느린 말속도 문장
        if stats["slow_speech"]["count"] > 0:
            print(f"\n느린 말속도 문장 수: {stats['slow_speech']['count']}개")
            segments_info = [f"문장 {seg[0]} ({seg[1]:.1f}~{seg[2]:.1f}초)" 
                           for seg in stats["slow_speech"]["segments"]]
            print(f"→ {', '.join(segments_info)}")
        
        # 빠른 말속도 문장
        if stats["fast_speech"]["count"] > 0:
            print(f"\n빠른 말속도 문장 수: {stats['fast_speech']['count']}개")
            segments_info = [f"문장 {seg[0]} ({seg[1]:.1f}~{seg[2]:.1f}초)" 
                           for seg in stats["fast_speech"]["segments"]]
            print(f"→ {', '.join(segments_info)}")
        
        # 불필요한 침묵 발생 문장
        if stats["silence_detected"]["count"] > 0:
            print(f"\n불필요한 침묵 발생 문장 수: {stats['silence_detected']['count']}개")
            segments_info = [f"문장 {seg[0]} ({seg[1]:.1f}~{seg[2]:.1f}초)" 
                           for seg in stats["silence_detected"]["segments"]]
            print(f"→ {', '.join(segments_info)}")
        
        # 말더듬이 감지된 문장
        if stats["stuttering_detected"]["count"] > 0:
            print(f"\n말더듬이 감지된 문장 수: {stats['stuttering_detected']['count']}개")
            segments_info = [f"문장 {seg[0]} ({seg[1]:.1f}~{seg[2]:.1f}초)" 
                           for seg in stats["stuttering_detected"]["segments"]]
            print(f"→ {', '.join(segments_info)}")
        
        # 음량 관련 통계
        if stats["volume_too_quiet"]["count"] > 0:
            print(f"\n📉 너무 작은 음량 문장 수: {stats['volume_too_quiet']['count']}개")
            segments_info = [f"문장 {seg[0]} ({seg[1]:.1f}~{seg[2]:.1f}초)" 
                           for seg in stats["volume_too_quiet"]["segments"]]
            print(f"→ {', '.join(segments_info)}")
        
        if stats["volume_too_loud"]["count"] > 0:
            print(f"\n📈 너무 큰 음량 문장 수: {stats['volume_too_loud']['count']}개")
            segments_info = [f"문장 {seg[0]} ({seg[1]:.1f}~{seg[2]:.1f}초)" 
                           for seg in stats["volume_too_loud"]["segments"]]
            print(f"→ {', '.join(segments_info)}")
        
        print(f"\n✅ 적절한 음량 문장 수: {stats['volume_good']['count']}개")
        
        print(f"\n📈 전체 평균:")
        print(f"평균 발음 점수: {stats['avg_pronunciation']:.1f}/100")
        if stats["avg_similarity"] > 0:
            print(f"평균 유사도 점수: {stats['avg_similarity']:.1f}/100")
        if stats["avg_speech_rate"] > 0:
            print(f"평균 말속도: {stats['avg_speech_rate']:.1f} WPM")
        print(f"📶 평균 RMS (음량): {stats['avg_rms']:.3f}")
        
        print("\n📝 종합 피드백:")
        overall_feedback = analyzer.generate_overall_feedback(stats)
        print(overall_feedback)
        
        # 전체 통계
        process_time = time.time() - start_process
        print(f"\n⌛ 총 처리 시간: {process_time:.1f}초")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        # 임시 파일 정리
        if 'ref_path' in locals() and os.path.exists(ref_path):
            os.remove(ref_path) 