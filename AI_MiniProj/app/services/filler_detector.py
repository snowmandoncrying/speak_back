import whisper
import torch
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import os
from abc import ABC, abstractmethod
from app.services.intonation_analyzer import analyze_intonation
from app.services.speed_analyzer import analyze_speed

# JAVA_HOME 설정 (KoNLPy 사용을 위해)
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'

class STTEngine(ABC):
    """STT 엔진 추상 클래스"""
    
    @abstractmethod
    def transcribe(self, file_path: str) -> Dict:
        pass

class WhisperSTT(STTEngine):
    """Whisper STT 엔진"""
    
    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self.model = None
        
    def load_model(self):
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔥 Using device: {device}")
            self.model = whisper.load_model(self.model_size, device=device)
        return self.model
    
    def transcribe(self, file_path: str) -> Dict:
        """음성 파일을 완전히 변환 (누락 없이)"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {file_path}")
        
        model = self.load_model()
        
        print(f"🎵 음성 파일 분석 중: {file_path}")
        print("⏳ 전체 분석을 위해 시간이 걸릴 수 있습니다...")
        
        # 말버릇 포함을 위한 최적화된 Whisper 설정
        result = model.transcribe(
            file_path,
            language="ko",
            task="transcribe",
            verbose=True,
            temperature=0.0,
            beam_size=5,
            best_of=5,
            patience=1.0,
            length_penalty=1.0,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            condition_on_previous_text=False,
            suppress_tokens=[-1],
            word_timestamps=True,
            initial_prompt=(
                "다음은 한국어 발표 음성입니다. '음', '어', '아', '그니까', '아마', '그래서' 등의 "
                "모든 말버릇과 감탄사를 포함하여 정확히 전사해주세요. 생략하지 마세요."
            )
        )
        
        print(f"✅ 분석 완료! {len(result.get('segments', []))}개 세그먼트 발견")
        return result

class VoskSTT(STTEngine):
    """Vosk STT 엔진 (말버릇 포함 가능)"""
    
    def __init__(self, model_path: str = None):
        try:
            import vosk
            self.vosk = vosk
            if model_path:
                self.model = vosk.Model(model_path)
            else:
                # 기본 한국어 모델 경로 (다운로드 필요)
                self.model = vosk.Model("models/vosk-model-ko-0.22")
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        except ImportError:
            raise ImportError("Vosk가 설치되지 않았습니다. pip install vosk로 설치하세요.")
    
    def transcribe(self, file_path: str) -> Dict:
        import wave
        import json
        
        segments = []
        
        # WAV 파일을 vosk로 처리
        wf = wave.open(file_path, 'rb')
        
        segment_id = 0
        current_text = ""
        start_time = 0
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
                
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                if result.get('text'):
                    segments.append({
                        'id': segment_id,
                        'start': start_time,
                        'end': start_time + (len(data) / 16000),
                        'text': result['text'],
                        'tokens': [],  # Vosk token 정보 추가 가능
                        'words': []
                    })
                    segment_id += 1
                    start_time += len(data) / 16000
        
        # 마지막 결과 처리
        final_result = json.loads(self.recognizer.FinalResult())
        if final_result.get('text'):
            segments.append({
                'id': segment_id,
                'start': start_time,
                'end': start_time + 0.5,
                'text': final_result['text'],
                'tokens': [],
                'words': []
            })
        
        wf.close()
        
        return {
            'segments': segments,
            'language': 'ko',
            'text': ' '.join([seg['text'] for seg in segments])
        }

class GoogleSTT(STTEngine):
    """Google Speech-to-Text API"""
    
    def __init__(self, credentials_path: str = None):
        try:
            from google.cloud import speech
            self.speech = speech
            if credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        except ImportError:
            raise ImportError("Google Cloud Speech API가 설치되지 않았습니다.")
    
    def transcribe(self, file_path: str) -> Dict:
        client = self.speech.SpeechClient()
        
        with open(file_path, 'rb') as audio_file:
            content = audio_file.read()
        
        audio = self.speech.RecognitionAudio(content=content)
        config = self.speech.RecognitionConfig(
            encoding=self.speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ko-KR",
            enable_word_time_offsets=True,
            enable_automatic_punctuation=False,  # 말버릇 보존위해 구두점 자동 추가 비활성화
            profanity_filter=False,  # 모든 단어 포함
            speech_contexts=[{
                'phrases': ['음', '어', '그니까', '아마', '그래서', '뭐냐면'],
                'boost': 20.0  # 말버릇 인식 강화
            }]
        )
        
        response = client.recognize(config=config, audio=audio)
        
        segments = []
        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]
            
            # 단어별 타임스탬프 처리
            words = []
            if hasattr(alternative, 'words'):
                for word_info in alternative.words:
                    words.append({
                        'word': word_info.word,
                        'start_time': word_info.start_time.total_seconds(),
                        'end_time': word_info.end_time.total_seconds()
                    })
            
            start_time = words[0]['start_time'] if words else 0
            end_time = words[-1]['end_time'] if words else 0
            
            segments.append({
                'id': i,
                'start': start_time,
                'end': end_time,
                'text': alternative.transcript,
                'confidence': alternative.confidence,
                'words': words,
                'tokens': []
            })
        
        return {
            'segments': segments,
            'language': 'ko',
            'text': ' '.join([seg['text'] for seg in segments])
        }

class FillerWordDetector:
    """완전한 말버릇 탐지 시스템"""
    
    def __init__(self, stt_engine: str = "whisper", model_size: str = "medium"):
        self.stt_engine_name = stt_engine
        
        # STT 엔진 초기화
        if stt_engine == "whisper":
            self.stt_engine = WhisperSTT(model_size)
        elif stt_engine == "vosk":
            self.stt_engine = VoskSTT()
        elif stt_engine == "google":
            self.stt_engine = GoogleSTT()
        else:
            raise ValueError(f"지원하지 않는 STT 엔진: {stt_engine}")
        
        # 말버릇 목록
        self.filler_words = [
            "음", "어", "그니까", "아마", "그래서", "뭐냐면", 
            "저기", "그게", "그러니까", "아", "네", "예"
        ]
        
        # 말버릇 패턴 (정규식)
        self.filler_patterns = [
            r'\b(음|어|그니까|아마|그래서|뭐냐면|저기|그게|그러니까)\b',
            r'(음{2,}|어{2,})',
            r'(음[.\s]{1,3}|어[.\s]{1,3})',
            r'(그[으음]*니까|그[으음]*래서)',
            r'(음\s*음|어\s*어|그니까\s*그니까)',
            r'그[.\s]*니까',
            r'아[.\s]*마',
            r'그[.\s]*래서',
            r'뭐[.\s]*냐면'
        ]
        
        self.confidence_threshold = 0.3
    
    def detect_fillers_in_text(self, text: str) -> Dict[str, int]:
        """텍스트에서 말버릇 탐지"""
        filler_counts = defaultdict(int)
        text_lower = text.lower()
        
        # 기본 단어 매칭
        for filler in self.filler_words:
            pattern = rf'\b{re.escape(filler)}\b'
            matches = re.findall(pattern, text_lower)
            filler_counts[filler] += len(matches)
        
        # 패턴 매칭
        for pattern in self.filler_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                matched_text = match.group()
                if any(filler in matched_text for filler in ["음", "ㅡ", "으"]):
                    filler_counts["음"] += 1
                elif any(filler in matched_text for filler in ["어", "ㅓ"]):
                    filler_counts["어"] += 1
                elif "니까" in matched_text:
                    filler_counts["그니까"] += 1
                elif "래서" in matched_text:
                    filler_counts["그래서"] += 1
                elif "마" in matched_text and "아" in matched_text:
                    filler_counts["아마"] += 1
                elif "냐면" in matched_text:
                    filler_counts["뭐냐면"] += 1
        
        return dict(filler_counts)
    
    def analyze_segments(self, segments: List[Dict]) -> List[Dict]:
        """세그먼트별 말버릇 분석"""
        analyzed_segments = []
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            
            # 말버릇 탐지
            filler_counts = self.detect_fillers_in_text(text)
            total_fillers = sum(filler_counts.values())
            
            # 토큰 정보 처리
            tokens = segment.get("tokens", [])
            words = segment.get("words", [])
            
            analyzed_segment = {
                "id": i + 1,
                "text": text,
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "duration": round(end_time - start_time, 2),
                "filler_counts": filler_counts,
                "total_fillers": total_fillers,
                "has_fillers": total_fillers > 0,
                "token_count": len(tokens),
                "word_count": len(words.split()) if isinstance(words, str) else len(words)
            }
            
            analyzed_segments.append(analyzed_segment)
        
        return analyzed_segments
    
    def detect_from_audio(self, file_path: str) -> Dict:
        """메인 분석 함수"""
        print(f"🔊 STT 엔진 사용: {self.stt_engine_name}")
        
        # STT 엔진으로 음성을 텍스트로 변환
        stt_result = self.stt_engine.transcribe(file_path)
        
        # 세그먼트 분석
        segments = stt_result.get("segments", [])
        analyzed_segments = self.analyze_segments(segments)
        
        # 전체 통계 계산
        total_stats = self.calculate_total_stats(analyzed_segments)
        
        # 결과 구성
        result = {
            "file_path": file_path,
            "stt_engine": self.stt_engine_name,
            "total_segments": len(analyzed_segments),
            "total_statistics": total_stats,
            "segments": analyzed_segments,
            "original_stt_result": stt_result
        }
        
        # 억양 분석
        intonation_results, avg_pitch_std, pitch_ranges = analyze_intonation(file_path, segments)
        # 속도 분석
        speed_results, avg_spm, avg_wpm = analyze_speed(file_path, segments)
        
        return result
    
    def calculate_total_stats(self, segments: List[Dict]) -> Dict:
        """전체 통계 계산"""
        total_filler_counts = defaultdict(int)
        segments_with_fillers = 0
        total_duration = 0
        
        for segment in segments:
            if segment["has_fillers"]:
                segments_with_fillers += 1
            
            for filler, count in segment["filler_counts"].items():
                total_filler_counts[filler] += count
            
            total_duration += segment["duration"]
        
        return {
            "total_fillers": sum(total_filler_counts.values()),
            "filler_breakdown": dict(total_filler_counts),
            "segments_with_fillers": segments_with_fillers,
            "filler_density": round(segments_with_fillers / len(segments) * 100, 2) if segments else 0,
            "total_duration": round(total_duration, 2)
        }
    
    def print_results(self, result: Dict):
        """결과를 콘솔에 출력"""
        print("\n" + "="*70)
        print("🎯 말버릇 탐지 결과")
        print("="*70)
        
        # 전체 통계
        stats = result["total_statistics"]
        print(f"\n📊 전체 통계:")
        print(f"  - 총 세그먼트 수: {result['total_segments']}개")
        print(f"  - 말버릇이 있는 세그먼트: {stats['segments_with_fillers']}개")
        print(f"  - 말버릇 빈도: {stats['filler_density']}%")
        print(f"  - 총 분석 시간: {stats['total_duration']}초")
        print(f"  - 총 말버릇 개수: {stats['total_fillers']}개")
        
        # 말버릇별 통계
        if stats['filler_breakdown']:
            print(f"\n🗣️ 말버릇별 통계:")
            for filler, count in sorted(stats['filler_breakdown'].items(), 
                                       key=lambda x: x[1], reverse=True):
                percentage = round(count / stats['total_fillers'] * 100, 1)
                print(f"  - {filler}: {count}회 ({percentage}%)")
        
        # 세그먼트별 결과
        print(f"\n📝 세그먼트별 분석:")
        print("-"*70)
        
        for segment in result["segments"]:
            print(f"\n[{segment['id']}] {segment['start']}s ~ {segment['end']}s")
            print(f"텍스트: {segment['text']}")
            
            if segment["has_fillers"]:
                print("말버릇:", end=" ")
                filler_strs = []
                for filler, count in segment["filler_counts"].items():
                    filler_strs.append(f"{filler}({count})")
                print(", ".join(filler_strs))
            else:
                print("말버릇: 없음")
        
        print("\n" + "="*70)
        print("✅ 분석 완료!")
        print("="*70)
    
    def save_to_json(self, result: Dict, output_path: str):
        """결과를 JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 결과가 저장되었습니다: {output_path}")

def main():
    """메인 실행 함수"""
    audio_file = "test_audio/b1055308.wav"
    
    try:
        print("🧪 Whisper STT 엔진 테스트")
        
        # 말버릇 탐지기 초기화
        detector = FillerWordDetector(stt_engine="whisper", model_size="medium")
        
        # 분석 실행
        result = detector.detect_from_audio(audio_file)
        
        # 결과 출력
        detector.print_results(result)
        
        # JSON 저장
        output_json = "filler_analysis_whisper.json"
        detector.save_to_json(result, output_json)
        
        # 억양 분석
        intonation_results, avg_pitch_std, pitch_ranges = analyze_intonation(audio_file, result["segments"])
        # 속도 분석
        speed_results, avg_spm, avg_wpm = analyze_speed(audio_file, result["segments"])
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()