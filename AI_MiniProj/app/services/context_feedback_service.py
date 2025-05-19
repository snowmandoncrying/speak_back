import google.generativeai as genai
import json, re

GEMINI_API_KEY = "AIzaSyBPN8mpN_L37VFeJ-6B7VKM8GWETbkVMYw"
genai.configure(api_key=GEMINI_API_KEY)

# 모델 객체 생성 (무료 사용 가능, 생략 시 기본 모델 사용)
model = genai.GenerativeModel("models/gemini-1.5-flash")

def get_segment_context_feedback(word: str) -> dict:
    prompt = (
        f"[{word}]: 이 문장에서 사용된 단어나 표현 중 어색하거나 과장되었거나 반복된 어휘가 있다면 해당 단어를 지적하고, 간단한 피드백을 한 문장으로 주세요. 문제가 없다면 '문제 없음'이라고만 답변하세요. 아래 JSON 형식으로만 답변하세요:\n"
        '{"vocabulary": ""}'
    )   
    response = model.generate_content(prompt)
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0).replace("'", '"'))
        except Exception:
            return {"vocabulary": "분석 불가"}
    return {"vocabulary": "분석 불가"}

def should_remove_vocabulary(value: str) -> bool:
    if value is None:
        return True
    v = value.strip().replace(" ", "")
    return v in ("", "분석불가", "문제없음") or value.strip() == "문제 없음"

def add_context_to_segments(segments: list) -> list:
    for seg in segments:
        word = seg.get("word") or seg.get("text") or ""
        context = get_segment_context_feedback(word)
        vocab = context.get("vocabulary", "")
        # context 키 제거, vocabulary만 최상위로 이동
        if "context" in seg:
            seg.pop("context")
        if not should_remove_vocabulary(vocab):
            seg["vocabulary"] = vocab
        elif "vocabulary" in seg:
            seg.pop("vocabulary")
    return segments
