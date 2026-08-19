import re
import json
import time
import random
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("Thiếu GROQ_API_KEY. Vui lòng thiết lập trong file .env hoặc biến môi trường.")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

def parse_json_object(text: str) -> dict:
    """
    Trích xuất và parse đối tượng JSON từ câu trả lời của LLM.
    """
    text = str(text).strip()
    # Loại bỏ thẻ think nếu có (từ qwen hoặc reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[a:b+1])

def groq_chat(messages: list, model: str = None, json_mode: bool = False, max_retries: int = 4):
    """
    Wrapper gọi Groq API kèm retry với exponential backoff và tự động fallback nếu gặp lỗi JSON validate hoặc rate limit.
    """
    client = get_groq_client()
    target_model = model or GROQ_MODEL or "openai/gpt-oss-20b"
    fallback_models = ["openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

    last_error = None
    curr_model = target_model

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": curr_model,
                "messages": messages,
                "temperature": 0.0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            return content, usage
        except Exception as e:
            last_error = e
            err_str = str(e)
            
            # Nếu gặp lỗi JSON validate, thử lại ngay mà không bật json_mode
            if "json_validate_failed" in err_str:
                json_mode = False
                continue
                
            if "429" in err_str or "rate_limit" in err_str.lower() or "404" in err_str or "decommissioned" in err_str:
                for fb in fallback_models:
                    if fb != curr_model:
                        print(f"[Groq Fallback] Chuyển model {curr_model} -> {fb}...")
                        curr_model = fb
                        break
            if attempt == max_retries - 1:
                break
            sleep_time = min(10.0, 2**attempt + random.random())
            time.sleep(sleep_time)

    raise RuntimeError(f"Groq chat failed sau {max_retries} lần thử: {last_error}")

def groq_json(system: str, user: str, model: str = None):
    """
    Tiện ích gửi system & user prompt và nhận lại đối tượng JSON được parse.
    """
    try:
        text, usage = groq_chat(
            [
                {"role": "system", "content": system + "\nOutput strict valid JSON only."},
                {"role": "user", "content": user}
            ],
            model=model,
            json_mode=True,
        )
        return parse_json_object(text), usage
    except Exception:
        # Fallback without json_mode
        text, usage = groq_chat(
            [
                {"role": "system", "content": system + "\nOutput strict valid JSON only."},
                {"role": "user", "content": user}
            ],
            model=model,
            json_mode=False,
        )
        return parse_json_object(text), usage
