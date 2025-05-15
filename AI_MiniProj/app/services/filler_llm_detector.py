import os
import openai
from dotenv import load_dotenv
import re
import json

# .env 파일에서 OPENAI_API_KEY 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 말버릇 분석용 프롬프트
FULL_TEXT_PROMPT = """다음은 발표 스크립트입니다. '음', '어', '그니까', '아마', '그래서', '뭐냐면', '저기', '그게', '그러니까', '아', '네', '예' 등의 말버릇이 포함된 문장만 골라 JSON으로 정리해주세요. 각 문장에 어떤 말버릇이 몇 번 등장했는지도 함께 표시해주세요. 

반드시 다음 형식으로 답변하세요:
[
  {{"문장": "음 저희는 이번에...", "말버릇": {{"음": 1}}}},
  {{"문장": "그니까 아마...", "말버릇": {{"그니까": 1, "아마": 1}}}}
]

텍스트:
{text}"""

def analyze_filler_from_text(full_text: str, verbose: bool = False) -> dict:
    """
    Whisper로 추출한 전체 텍스트를 받아 LLM(말버릇) 분석만 수행합니다.
    Args:
        full_text (str): Whisper로부터 얻은 전체 텍스트
        verbose (bool): 상세 출력 여부
    Returns:
        dict: LLM 분석 결과 (filler_sentences 등 포함)
    """
    prompt = FULL_TEXT_PROMPT.format(text=full_text)
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000
        )
        content = response.choices[0].message.content.strip()
        # JSON 추출 및 파싱
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            filler_sentences = json.loads(json_str)
            # 전체 통계 계산
            total_filler_counts = {}
            total_fillers = 0
            for sentence_data in filler_sentences:
                if "말버릇" in sentence_data:
                    for filler, count in sentence_data["말버릇"].items():
                        total_filler_counts[filler] = total_filler_counts.get(filler, 0) + count
                        total_fillers += count
            if verbose:
                print(" 말버릇 분석 완료!")
            return {
                "success": True,
                "full_text": full_text,
                "filler_sentences": filler_sentences,
                "total_filler_counts": total_filler_counts,
                "total_fillers": total_fillers,
                "total_sentences_with_fillers": len(filler_sentences)
            }
        else:
            return {
                "success": False,
                "error": "LLM 응답에서 JSON 형식을 찾을 수 없음",
                "full_text": full_text,
                "filler_sentences": [],
                "total_filler_counts": {},
                "total_fillers": 0,
                "total_sentences_with_fillers": 0
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"LLM 분석 오류: {str(e)}",
            "full_text": full_text,
            "filler_sentences": [],
            "total_filler_counts": {},
            "total_fillers": 0,
            "total_sentences_with_fillers": 0
        }

def build_filler_map_from_result(filler_result: dict, whisper_segments: list) -> dict[int, str]:
    """
    whisper의 segment 목록과 말버릇 분석 결과(filler_result)를 연결하여,
    segment.id별로 해당 문장에 포함된 말버릇(필러) 문자열을 반환하는 맵을 생성합니다.
    Args:
        filler_result (dict): analyze_filler_from_text()의 리턴값 ("filler_sentences" 포함)
        whisper_segments (list): Whisper의 segment 리스트 (각 segment는 'id', 'text' 등 포함)
    Returns:
        dict[int, str]: segment.id별로 필러(말버릇) 문자열 (없으면 '없음')
    """
    def normalize(text):
        return re.sub(r'\s+', '', re.sub(r'[.,!?…~\-]', '', text)).strip()

    filler_map = {}
    filler_sentences = filler_result.get("filler_sentences", [])
    used = set()

    for seg in whisper_segments:
        seg_id = seg.get("id")
        seg_text = normalize(seg.get("text", ""))
        found = False
        for idx, fs in enumerate(filler_sentences):
            fs_text = normalize(fs.get("문장", ""))
            if fs_text and fs_text in seg_text and idx not in used:
                fillers = fs.get("말버릇", {})
                filler_str = ", ".join(fillers.keys()) if fillers else "없음"
                filler_map[seg_id] = filler_str
                used.add(idx)
                found = True
                break
        if not found:
            filler_map[seg_id] = "없음"
    return filler_map