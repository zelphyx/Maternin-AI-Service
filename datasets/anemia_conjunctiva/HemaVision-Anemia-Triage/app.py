"""
HemaVision — Uncertainty-Aware Anemia Triage Assistant

Gradio demo app for Hugging Face Spaces.

Combines:
  1. MobileNetV2 + MC Dropout (uncertainty-aware image prediction)
  2. Fireworks AI (gpt-oss-20b) for medical reasoning / triage explanation


IMPORTANT — Screening tool disclaimer:
This tool estimates anemia likelihood from a conjunctival pallor image and is
intended for early screening and education only. It is NOT a diagnosis and
does not replace laboratory hemoglobin testing.
"""

import os
import io
import json
import requests
import numpy as np
import gradio as gr
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# CONFIG

MODEL_PATH = os.environ.get("MODEL_PATH", "mobilenet_final.keras")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODEL = "accounts/fireworks/models/gpt-oss-20b"

# Calibrated on a small validation sample (75th percentile of MC Dropout std_dev).
# NOTE: recalibrate on a larger validation set before production use.
UNCERTAINTY_STD_THRESHOLD = 0.095
MC_DROPOUT_ITERATIONS = 30


# MODEL LOAD (once, at app startup)

print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")



# MC Dropout uncertainty estimation

def preprocess_pil_image(pil_image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = pil_image.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def mc_dropout_predict(model, image_array, n_iterations=MC_DROPOUT_ITERATIONS):
    predictions = []
    for _ in range(n_iterations):
        pred = model(image_array, training=True)  # forces dropout active
        predictions.append(float(pred.numpy().flatten()[0]))
    predictions = np.array(predictions)
    return float(predictions.mean()), float(predictions.std())


def classify_with_uncertainty(mean_prob, std_dev, std_threshold=UNCERTAINTY_STD_THRESHOLD):
    is_uncertain = std_dev > std_threshold

    if mean_prob < 0.35:
        risk = "Low"
    elif mean_prob < 0.65:
        risk = "Moderate"
    else:
        risk = "High"

    return {
        "predicted_probability": round(mean_prob, 4),
        "uncertainty_std": round(std_dev, 4),
        "confident_prediction": not is_uncertain,
        "risk_tier": risk,
    }



# LLM reasoning layer (Fireworks / gpt-oss-20b)

def get_llm_reasoning(model_result: dict, age: str, gender: str, symptoms: str) -> dict:
    if not FIREWORKS_API_KEY:
        return {
            "risk_level": model_result["risk_tier"],
            "explanation": "LLM reasoning unavailable — FIREWORKS_API_KEY not configured.",
            "recommended_tests": ["Complete Blood Count (CBC)"],
            "nutrition_advice": "N/A",
            "referral_urgency": "moderate",
            "disclaimer": "This is a screening tool, not a medical diagnosis.",
        }

    system_prompt = (
        "You are a medical triage assistant supporting a non-invasive anemia "
        "screening tool. This tool is for SCREENING and EDUCATION only — it is "
        "NOT a diagnosis and must not replace laboratory hemoglobin testing. "
        "Respond ONLY with a valid JSON object, no other text, no reasoning shown."
    )

    user_prompt = f"""
    Image-based anemia screening result (MobileNetV2 + MC Dropout uncertainty):
    - predicted_probability: {model_result['predicted_probability']} (probability of anemia, 0-1 scale)
    - uncertainty_std: {model_result['uncertainty_std']} (std dev across 30 stochastic forward passes)
    - confident_prediction: {model_result['confident_prediction']}
    - risk_tier: {model_result['risk_tier']}

    Patient-reported context:
    - age: {age or "not provided"}
    - gender: {gender or "not provided"}
    - symptoms: {symptoms or "none reported"}

    Return a JSON object with exactly these keys:
    - risk_level (string: Low/Moderate/High)
    - explanation (2-3 sentences, plain simple language, non-diagnostic tone)
    - recommended_tests (array of strings, e.g. CBC, ferritin)
    - nutrition_advice (1-2 sentences, general iron-rich diet guidance)
    - referral_urgency (string: low/moderate/urgent)
    - disclaimer (string: a short reminder this is a screening tool, not a diagnosis)

    If confident_prediction is false, explicitly mention in the explanation that
    the model itself is uncertain about this image and a repeat test or clinical
    exam is recommended regardless of the risk tier.
    """

    payload = {
        "model": FIREWORKS_MODEL,
        "max_tokens": 1000,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(FIREWORKS_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            cleaned = raw_content.strip().strip("```json").strip("```").strip()
            return json.loads(cleaned)

    except Exception as e:
        return {
            "risk_level": model_result["risk_tier"],
            "explanation": f"LLM reasoning temporarily unavailable ({e}). Showing model-only result.",
            "recommended_tests": ["Complete Blood Count (CBC)"],
            "nutrition_advice": "N/A",
            "referral_urgency": "moderate",
            "disclaimer": "This is a screening tool, not a medical diagnosis.",
        }



# GRADIO INTERFACE

def format_report(model_result: dict, llm_result: dict) -> str:
    confidence_note = (
        " Model is confident in this prediction."
        if model_result["confident_prediction"]
        else " Model uncertainty is HIGH for this image — a repeat test or clinical exam is recommended regardless of risk level."
    )

    tests = "\n".join(f"  - {t}" for t in llm_result.get("recommended_tests", []))

    report = f"""
##  Triage Report

**Risk Level:** {llm_result.get('risk_level', model_result['risk_tier'])}
**Model Probability (anemic):** {model_result['predicted_probability']}
**Uncertainty (std dev, 30 passes):** {model_result['uncertainty_std']}

{confidence_note}

### Explanation
{llm_result.get('explanation', 'N/A')}

### Recommended Tests
{tests}

### Nutrition Advice
{llm_result.get('nutrition_advice', 'N/A')}

### Referral Urgency
**{llm_result.get('referral_urgency', 'N/A').upper()}**

---
_{llm_result.get('disclaimer', 'This is a screening tool, not a medical diagnosis.')}_
"""
    return report


def analyze_image(image: Image.Image, age: str, gender: str, symptoms: str):
    if image is None:
        return "Please upload a conjunctiva (lower eyelid) image to begin."

    image_array = preprocess_pil_image(image)
    mean_prob, std_dev = mc_dropout_predict(model, image_array)
    model_result = classify_with_uncertainty(mean_prob, std_dev)
    llm_result = get_llm_reasoning(model_result, age, gender, symptoms)

    return format_report(model_result, llm_result)


with gr.Blocks(title="HemaVision - Anemia Triage Assistant") as demo:
    gr.Markdown(
        """
        #  HemaVision - Uncertainty-Aware Anemia Triage Assistant
        Upload a conjunctiva (lower eyelid) image for a non-invasive anemia risk screening.

        **This tool is for screening and education only. It is NOT a diagnosis
        and does not replace laboratory hemoglobin testing.**
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Conjunctiva Image")
            age_input = gr.Textbox(label="Age (optional)", placeholder="e.g. 25")
            gender_input = gr.Textbox(label="Gender (optional)", placeholder="e.g. female")
            symptoms_input = gr.Textbox(
                label="Symptoms (optional, comma-separated)",
                placeholder="e.g. fatigue, dizziness, shortness of breath",
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            output_report = gr.Markdown(label="Triage Report")

    submit_btn.click(
        fn=analyze_image,
        inputs=[image_input, age_input, gender_input, symptoms_input],
        outputs=output_report,
    )

if __name__ == "__main__":
    demo.launch()
