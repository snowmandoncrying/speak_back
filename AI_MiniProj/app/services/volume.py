import librosa
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

class PronunciationAnalyzer:
    def validate_audio_file(self, file_path: str) -> bool:
        """오디오 파일 유효성 검사"""
        try:
            if not os.path.exists(file_path):
                return False
            
            y, sr = librosa.load(file_path, sr=None, duration=0.1)
            return len(y) > 0 and sr > 0
        
        except Exception:
            return False

    def load_audio_segment(self, audio_path: str, start_time: float = None, end_time: float = None) -> tuple:
        """
        오디오 파일을 로드하고 지정된 구간을 추출합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            start_time: 시작 시간 (초)
            end_time: 종료 시간 (초)
            
        Returns:
            tuple: (오디오 데이터, 샘플링 레이트) 또는 오류 시 (None, None)
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            if start_time is not None and end_time is not None:
                if start_time >= len(y)/sr or end_time > len(y)/sr:
                    print(f"⚠️ 요청한 구간이 오디오 길이를 초과합니다: {start_time:.1f}~{end_time:.1f}초 (전체 길이: {len(y)/sr:.1f}초)")
                    return None, None
                
                start_idx = int(start_time * sr)
                end_idx = int(end_time * sr)
                y = y[start_idx:end_idx]
            
            if len(y) == 0:
                print("⚠️ 추출된 오디오 구간이 비어있습니다.")
                return None, None
            
            return y, sr
            
        except Exception as e:
            print(f"❌ 오디오 로드 오류: {e}")
            return None, None

    def analyze_volume(self, audio_path: str, start_time: float, end_time: float) -> dict:
        """음량(RMS) 분석"""
        try:
            # 1. 오디오 로드 및 구간 추출
            y, sr = self.load_audio_segment(audio_path, start_time, end_time)
            if y is None:
                return {
                    "rms": 0.0,
                    "status": "분석 불가",
                    "feedback": "오디오 데이터를 로드할 수 없습니다."
                }

            # 2. RMS 에너지 계산 (프레임 단위)
            frame_length = 2048  # 약 0.1초 단위
            hop_length = 512     # 프레임 간 이동 간격
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 3. 유효한 음성 구간의 RMS만 선택
            energy_threshold = 0.005
            valid_rms = rms[rms > energy_threshold]
            
            if len(valid_rms) == 0:
                mean_rms = 0.0
            else:
                mean_rms = np.percentile(valid_rms, 80)

            # 4. 음량 등급 분류 (세분화된 기준)
            if mean_rms < 0.02:
                volume_status = "Too Quiet"
                feedback = "목소리가 작습니다. 조금 더 힘 있게 말해보세요."
            elif mean_rms < 0.04:
                volume_status = "조금 작음"
                feedback = "전달력은 있으나 약간 더 크게 말하면 좋습니다."
            elif mean_rms <= 0.065:
                volume_status = "적절"
                feedback = "목소리 크기가 적절하여 발표 전달력이 좋습니다."
            elif mean_rms <= 0.08:
                volume_status = "조금 큼"
                feedback = "전달력은 좋지만 약간만 낮춰도 좋습니다."
            else:
                volume_status = "Too Loud"
                feedback = "목소리가 다소 커서 부담스러울 수 있습니다. 약간 낮춰보세요."

            print(f"[DEBUG] mean_rms: {mean_rms:.5f}, 분류결과: {volume_status}")

            return {
            "rms": float(mean_rms),
            "status": volume_status,
            "feedback": feedback
            }

        except Exception as e:
            print(f"❌ 음량 분석 오류: {e}")
            return {
                "rms": 0.0,
                "status": "분석 불가",
                "feedback": f"음량 분석 중 오류가 발생했습니다: {str(e)}",
            }
# 침묵 분석 로직
    def detect_silence_segments(self, audio_path: str, top_db: int = 30, min_silence_gap: float = 0.1) -> list:
        """
        librosa를 사용하여 오디오 파일에서 무음 구간을 탐지합니다.
        민감도를 높여 더 작은 소리와 짧은 간격도 감지합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            top_db: 무음 감지를 위한 데시벨 임계값 (기본값: 30, 낮을수록 민감)
            min_silence_gap: 무음으로 인정할 최소 간격 (초) (기본값: 0.1초)
            
        Returns:
            list: 무음 구간 정보를 담은 딕셔너리 리스트
        """
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=None)
            
            # 유효 발화 구간 탐지 (더 민감한 설정)
            intervals = librosa.effects.split(y, 
                                           top_db=top_db,
                                           frame_length=1024,  # 더 작은 프레임 사용
                                           hop_length=256)    # 더 조밀한 분석
            
            # 무음 구간 추출
            silence_segments = []
            prev_end = 0
            
            for start, end in intervals:
                silence_start = prev_end / sr
                silence_end = start / sr
                duration = silence_end - silence_start
                
                if duration >= min_silence_gap:
                    # 무음 구간의 평균 에너지 계산
                    segment = y[prev_end:start]
                    if len(segment) > 0:
                        rms = np.sqrt(np.mean(segment**2))
                        
                        silence_segments.append({
                            "start": silence_start,
                            "end": silence_end,
                            "duration": duration,
                            "rms": float(rms)  # 무음 구간의 실제 에너지 레벨
                        })
                prev_end = end
                
            # 마지막 구간 이후 무음이 있다면 추가
            if prev_end < len(y):
                silence_start = prev_end / sr
                silence_end = len(y) / sr
                duration = silence_end - silence_start
                
                if duration >= min_silence_gap:
                    segment = y[prev_end:]
                    if len(segment) > 0:
                        rms = np.sqrt(np.mean(segment**2))
                        
                        silence_segments.append({
                            "start": silence_start,
                            "end": silence_end,
                            "duration": duration,
                            "rms": float(rms)
                        })
                
            return silence_segments
            
        except Exception as e:
            print(f"❌ 무음 구간 탐지 오류: {e}")
            return []

    def analyze_silence(self, current_segment, previous_segment, silence_segments: list = None) -> tuple:
        """
        두 발화 구간 사이의 침묵을 분석하여 피드백을 제공합니다.
        더 짧은 침묵도 감지하여 분석합니다.
        
        Args:
            current_segment: 현재 발화 구간 정보
            previous_segment: 이전 발화 구간 정보
            silence_segments: librosa로 탐지한 무음 구간 리스트 (선택)
            
        Returns:
            tuple: (침묵 길이, 피드백 메시지) 또는 (None, None)
        """
        try:
            if not previous_segment:
                return None, None
            
            # Whisper 기반 침묵 길이 계산
            whisper_silence = current_segment["start"] - previous_segment["end"]
            
            # librosa 기반 침묵 길이 확인 (제공된 경우)
            waveform_silence = 0
            if silence_segments:
                for seg in silence_segments:
                    # 현재 구간과 이전 구간 사이에 있는 무음 찾기
                    if (seg["start"] >= previous_segment["end"] and 
                        seg["end"] <= current_segment["start"]):
                        # RMS 값이 매우 낮은 경우에만 실제 침묵으로 간주
                        if seg["rms"] < 0.01:
                            waveform_silence = max(waveform_silence, seg["duration"])
            
            # 두 방식의 침묵 길이 중 더 큰 값 사용
            silence_duration = max(whisper_silence, waveform_silence)

            # 완화된 기준 적용 (더 짧은 침묵도 포함)
            if 0.1 <= silence_duration < 0.5:
                feedback = "자연스러운 호흡 간격입니다."
            elif 0.5 <= silence_duration < 1.5:
                feedback = "적절한 멈춤으로 청자가 내용을 이해하기 좋습니다."
            elif 1.5 <= silence_duration < 2.5:
                feedback = "다소 긴 멈춤이지만 강조 효과가 있을 수 있습니다."
            elif silence_duration >= 2.5:
                feedback = "침묵이 길어 청자의 집중이 끊길 수 있습니다."
            else:
                return None, None  # 0.1초 미만은 피드백 제외

            return float(silence_duration), feedback

        except Exception as e:
            print(f"❌ 침묵 감지 오류: {e}")
            return None, None

    def analyze_speech(self, audio_path: str, segments: list) -> dict:
        """음성 분석 메인 함수"""
        try:
            # 1. 입력 검증
            if not segments:
                return {
                    "sentence_feedback": []
                }

            # 2. librosa 기반 무음 구간 탐지
            silence_segments = self.detect_silence_segments(audio_path)

            # 3. 문장별 분석
            sentence_feedback = []
            previous_segment = None

            for segment in segments:
                # 3.1 음량 분석
                volume = self.analyze_volume(
                    audio_path,
                    segment["start"],
                    segment["end"]
                )

                # 3.2 침묵 분석 (librosa 결과 포함)
                silence_duration, silence_feedback = self.analyze_silence(
                    segment, 
                    previous_segment,
                    silence_segments
                )

                # 3.3 결과 구성
                feedback_entry = {
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                }

                if volume:
                    feedback_entry["volume"] = volume

                if silence_feedback:
                    feedback_entry["silence"] = {
                        "duration": silence_duration,
                        "feedback": silence_feedback
                    }

                sentence_feedback.append(feedback_entry)
                previous_segment = segment
                
            # 4. 전체 침묵 분석
            overall_silence = self.analyze_overall_silence(audio_path, segments)

            return {
                "sentence_feedback": sentence_feedback,
                "overall_silence": overall_silence,
                "silence_segments": silence_segments
            }
        
        except Exception as e:
            return {
                "sentence_feedback": [],
                "overall_silence": None,
                "silence_segments": []
            }

    def analyze_overall_silence(self, audio_path: str, segments: list) -> dict:
        """
        전체 오디오에서 침묵 비율을 분석하여 피드백을 제공합니다.
        더 정밀한 침묵 감지를 적용합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            segments: 발화 구간 정보 리스트
            
        Returns:
            dict: 전체 침묵 분석 결과
        """
        try:
            # 1. 총 오디오 길이 (초)
            total_duration = librosa.get_duration(filename=audio_path)
            
            # 2. Whisper segments 기반 침묵 시간 합산
            whisper_silence = 0.0
            for i in range(1, len(segments)):
                silence = segments[i]["start"] - segments[i-1]["end"]
                if silence > 0.1:  # 0.1초 이상의 침묵만 포함
                    whisper_silence += silence
                    
            # 3. librosa 기반 무음 구간 탐지 (더 민감한 설정)
            silence_segments = self.detect_silence_segments(audio_path, top_db=30, min_silence_gap=0.1)
            
            # 실제 무음인 구간만 선택 (RMS 기준)
            valid_segments = [seg for seg in silence_segments if seg["rms"] < 0.01]
            waveform_silence = sum(seg["duration"] for seg in valid_segments)
            
            # 4. 두 방식의 침묵 시간 중 더 큰 값 사용
            total_silence = max(whisper_silence, waveform_silence)
                
            # 5. 침묵 비율 계산
            silence_ratio = (total_silence / total_duration) * 100  # 퍼센트로 변환
            
            # 6. 침묵 비율에 따른 피드백 (세분화된 기준)
            if silence_ratio < 5:
                feedback = "침묵이 매우 적어 호흡이 빠른 발표입니다. 여유를 가져보세요."
            elif 5 <= silence_ratio < 10:
                feedback = "침묵이 다소 적지만 발표 속도가 적절합니다."
            elif 10 <= silence_ratio <= 20:
                feedback = "자연스러운 침묵 비율로 청자가 내용을 이해하기 좋습니다."
            elif 20 < silence_ratio <= 25:
                feedback = "침묵이 다소 많지만 강조 효과가 있을 수 있습니다."
            else:
                feedback = "침묵이 많아 발표의 흐름이 끊길 수 있습니다."
                
            return {
                "total_silence": float(total_silence),
                "total_duration": float(total_duration),
                "silence_ratio": float(silence_ratio),
                "feedback": feedback,
                "whisper_silence": float(whisper_silence),
                "waveform_silence": float(waveform_silence),
                "silence_segments": valid_segments  # RMS 기준으로 필터링된 구간만 반환
            }
            
        except Exception as e:
            print(f"❌ 전체 침묵 분석 오류: {e}")
            return None 