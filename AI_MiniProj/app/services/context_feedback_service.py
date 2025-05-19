import google.generativeai as genai
import asyncio
import json, re

GEMINI_API_KEY = "AIzaSyBPN8mpN_L37VFeJ-6B7VKM8GWETbkVMYw"
genai.configure(api_key=GEMINI_API_KEY)

# 모델 객체 생성 (무료 사용 가능, 생략 시 기본 모델 사용)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# 긍정/당연한 coherence 표현 리스트
POSITIVE_COHERENCE_PHRASES = [
    "논리 흐름이 자연스럽습니다",
    "자연스럽게 연결됩니다",
    "문장의 논리적 흐름은 자연스럽습니다",
    "전체적으로 의미 전달이 잘 됩니다",
    "자연스럽고 적절합니다"
]

def is_positive_coherence(text: str) -> bool:
    if not text:
        return False
    for phrase in POSITIVE_COHERENCE_PHRASES:
        if phrase in text:
            return True
    return False

async def get_segment_context_feedback(word: str) -> dict:
    prompt = (
        f"[{word}]: 이 문장의 논리 흐름(coherence)만 한글로 평가해서 아래 JSON 형식으로 답변하세요.\n"
        '{"coherence": ""}'
    )
    response = await asyncio.to_thread(model.generate_content, prompt)
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {
        "coherence": "분석 불가"
    }

async def add_context_to_segments(segments: list) -> list:
    tasks = []
    for seg in segments:
        word = seg.get("word") or seg.get("text") or ""
        tasks.append(get_segment_context_feedback(word))
    context_results = await asyncio.gather(*tasks)
    for seg, context in zip(segments, context_results):
        # 긍정/당연한 coherence면 coherence 키 삭제
        coherence = context.get("coherence")
        if is_positive_coherence(coherence):
            context.pop("coherence", None)
        seg["context"] = context
    return segments