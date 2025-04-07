# backend/query_llm.py

import os
import gradio as gr
from typing import Any, Dict, Generator, List
from openai import OpenAI
from huggingface_hub import InferenceClient

# ✅ Load environment variables
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
HF_MODEL = os.getenv("HF_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()

# ✅ Initialize clients
OAI_CLIENT = OpenAI()
HF_CLIENT = InferenceClient(HF_MODEL, token=HF_TOKEN)

# ✅ Generation configs
HF_GENERATE_KWARGS = {
    "temperature": max(float(os.getenv("TEMPERATURE", 0.9)), 1e-2),
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", 256)),
    "top_p": float(os.getenv("TOP_P", 0.6)),
    "repetition_penalty": float(os.getenv("REP_PENALTY", 1.2)),
    "do_sample": os.getenv("DO_SAMPLE", "true").lower() == "true",
}

OAI_GENERATE_KWARGS = {
    "temperature": max(float(os.getenv("TEMPERATURE", 0.9)), 1e-2),
    "max_tokens": int(os.getenv("MAX_NEW_TOKENS", 256)),
    "top_p": float(os.getenv("TOP_P", 0.6)),
    "frequency_penalty": max(-2, min(float(os.getenv("FREQ_PENALTY", 0)), 2)),
}

def format_prompt(message: str, api_kind: str):
    """Format prompt depending on backend."""
    if api_kind == "openai":
        return [{"role": "user", "content": message}]
    elif api_kind == "hf":
        return message
    else:
        raise ValueError(f"Unsupported API kind: {api_kind}")

def generate_hf(prompt: str, history: str = "") -> Generator[str, None, str]:
    """Stream Hugging Face Inference output."""
    try:
        stream = HF_CLIENT.text_generation(
            prompt,
            stream=True,
            details=True,
            return_full_text=False,
            **HF_GENERATE_KWARGS,
        )
        output = ""
        for response in stream:
            output += response.token.text
            yield output
    except Exception as e:
        raise gr.Error(f"Hugging Face error: {str(e)}")

def generate_openai(prompt: str, history: str = "") -> Generator[str, None, str]:
    """Stream OpenAI output."""
    try:
        stream = OAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=format_prompt(prompt, "openai"),
            stream=True,
            **OAI_GENERATE_KWARGS,
        )
        output = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                output += chunk.choices[0].delta.content
                yield output
    except Exception as e:
        raise gr.Error(f"OpenAI error: {str(e)}")

def answer_question(prompt: str, history: str = "") -> str:
    """Unified LLM answer function."""
    generator = generate_hf(prompt, history) if LLM_BACKEND == "hf" else generate_openai(prompt, history)
    final_response = ""
    for token in generator:
        final_response = token
    return final_response
