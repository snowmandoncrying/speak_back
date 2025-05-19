import os
import openai
from dotenv import load_dotenv
from typing import Dict, List
import json

# .env 파일에서 OPENAI_API_KEY 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_qa_pairs(text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    입력된 텍스트를 기반으로 3~5개의 질문과 답변 쌍을 생성합니다.
    
    Args:
        text (str): 질문과 답변을 생성할 원본 텍스트
        
    Returns:
        Dict[str, List[Dict[str, str]]]: 질문과 답변 쌍의 리스트를 포함하는 딕셔너리
    """
    prompt = f"""
    다음 텍스트를 바탕으로 3~5개의 관련 질문과 답변을 생성해주세요.
    반드시 아래 JSON 형식으로만 응답해주세요:

    [
        {{"question": "질문1", "answer": "답변1"}},
        {{"question": "질문2", "answer": "답변2"}},
        ...
    ]

    - 질문은 텍스트의 핵심 내용을 파악할 수 있도록 구성해주세요.
    - 답변은 질문에 대한 포괄적이고 명확한 설명을 제공해주세요.
    - 질문과 답변은 모두 한국어로 작성해주세요.

    텍스트:
    {text}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 주어진 텍스트를 바탕으로 질문과 답변을 생성하는 전문가입니다. 반드시 지정된 JSON 형식으로만 응답해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract the response content
        content = response.choices[0].message.content.strip()
        
        try:
            # JSON 파싱 시도
            qa_pairs = json.loads(content)
            return {"questions_and_answers": qa_pairs}
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 전체 내용을 하나의 Q&A로 반환
            return {
                "questions_and_answers": [
                    {
                        "question": "원본 응답",
                        "answer": content
                    }
                ]
            }
        
    except Exception as e:
        raise Exception(f"Q&A 생성 중 오류 발생: {str(e)}") 