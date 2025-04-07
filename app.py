import os
import logging
from pathlib import Path
from time import perf_counter

import gradio as gr
from jinja2 import Environment, FileSystemLoader

from backend.image_ocr import extract_text_from_image
from backend.query_llm import generate_hf, generate_openai, answer_question
from backend.semantic_search import retrieve

# ✅ Environment setup
TOP_K = int(os.getenv("TOP_K", 4))
proj_dir = Path(__file__).parent

# ✅ Logging setup
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# ✅ Jinja2 templates
env = Environment(loader=FileSystemLoader(proj_dir / "templates"))

# --- Logic Functions ---

def add_text(history, text):
    """Add user input to chat history."""
    history = history or []
    history.append((text, None))
    return history, gr.Textbox(value="", interactive=False)

def bot(history, api_kind):
    """Handle text chat: retrieve documents and generate LLM response."""
    query = history[-1][0]
    if not query:
        raise gr.Warning("Please submit a non-empty string as a prompt.")

    logger.info("🔎 Retrieving documents...")
    start_time = perf_counter()
    documents = retrieve(query, TOP_K)
    logger.info(f"✅ Retrieved {len(documents)} documents in {round(perf_counter() - start_time, 2)}s")

    # Render prompt templates
    prompt = env.get_template("template.j2").render(documents=documents, query=query)
    prompt_html = env.get_template("template_html.j2").render(documents=documents, query=query)

    generate_fn = generate_hf if api_kind == "HuggingFace" else generate_openai
    history[-1][1] = ""
    for character in generate_fn(prompt, history[:-1]):
        history[-1][1] = character
        yield history, prompt_html

def extract_and_respond(image, api_kind):
    """Handle image upload: EasyOCR text extraction and LLM answer."""
    extracted_text = extract_text_from_image(image)

    logger.info(f"[DEBUG] OCR Text Extracted:\n{extracted_text}\n")

    if not extracted_text.strip():
        return "⚠️ No readable text found in the uploaded image.", ""

    final_answer = answer_question(extracted_text)
    return extracted_text, final_answer

# --- UI Layout ---

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 DocuFlow\n**Upload text or images and get AI-powered answers from your engineering documents.**")

    with gr.Tab("💬 Text Chat"):
        chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Ask something...", scale=3)
            txt_btn = gr.Button("Submit", scale=1)
        api_kind = gr.Radio(choices=["HuggingFace", "OpenAI"], value="HuggingFace", label="Choose LLM")
        prompt_html = gr.HTML()

        txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False) \
               .then(bot, [chatbot, api_kind], [chatbot, prompt_html]) \
               .then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    with gr.Tab("🖼️ Image Upload"):
        img = gr.Image(type="pil", label="Upload an image (requirements, documents, diagrams)")
        img_api_kind = gr.Radio(choices=["HuggingFace", "OpenAI"], value="HuggingFace", label="Choose LLM Backend")
        with gr.Row():
            img_ocr_text = gr.Textbox(label="🔍 Extracted OCR Text", lines=6, interactive=False)
            img_output = gr.Textbox(label="🧠 LLM Answer")

        gr.Button("Extract & Analyze").click(
            fn=extract_and_respond,
            inputs=[img, img_api_kind],
            outputs=[img_ocr_text, img_output]
        )

demo.queue()
demo.launch(debug=True)
