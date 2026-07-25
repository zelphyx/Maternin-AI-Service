#  HemaVision 
# Uncertainty-Aware Anemia Triage Assistant
  AMD Developer Hackathon: ACT II - Unicorn Track
## Project Objective

HemaVision is a non-invasive anemia screening assistant that goes beyond a single confidence number. By combining a MobileNetV2 image classifier with Monte Carlo Dropout uncertainty quantification and an LLM-based medical reasoning layer, the system communicates not just *what* it predicts, but *how confident* it is , the difference between a demo and a tool that could be responsibly deployed.

The core finding this project is built on: in our own peer-reviewed benchmarking study on this exact task, MobileNetV2's 5-fold cross-validation accuracy on conjunctival pallor images was **62.5% ± 8.1%**, with single-split evaluation inflating results by up to 10 percentage points. Rather than hide that variability behind one confident-sounding number, HemaVision exposes it , per prediction, in real time  and explicitly flags results it isn't sure about.

---

## Live Demo

**[ Open Live Demo →](https://huggingface.co/spaces/AminahAsif/HemaVision-Anemia-Triage)**

Upload a conjunctiva (lower eyelid) image, optionally add age, gender, and symptoms, and get: an anemia risk prediction, a calibrated uncertainty score, a plain-language explanation, recommended lab tests, nutrition guidance, and referral urgency — with an explicit warning when the model itself is uncertain.

 Full walkthrough: [`docs/Demo_Video.mp4`](docs/Demo_Video.mp4)
 Pitch deck: [`docs/Pitch_Deck.pptx`](docs/HemaVision_Pitch_Deck.pptx)

---

## Key Results

| Metric | Result |
|---|---|
| MobileNetV2 single-split accuracy | 72.5% |
| MobileNetV2 5-fold CV accuracy | **62.5% ± 8.1%** |
| Random Forest 5-fold CV accuracy (best classical baseline) | 74.5% ± 2.1% |
| MC Dropout iterations per prediction | 30 stochastic forward passes |
| Uncertainty threshold (calibrated) | 0.095 (75th percentile of validation std dev) |
| Reasoning layer | gpt-oss-20b via Fireworks AI, on AMD Instinct MI300X |


---

## How It Works

```
Conjunctival Image + Symptoms
            │
            ▼
  MobileNetV2 + Monte Carlo Dropout
  (30 stochastic forward passes)
            │
            ▼
  Prediction + Uncertainty (std dev)
            │
            ▼
  gpt-oss-20b (Fireworks AI / AMD MI300X)
  Medical reasoning layer
            │
            ▼
  Triage Report:
  risk level · explanation · recommended tests
  nutrition guidance · referral urgency
   uncertainty warning (when applicable)
```

**Why uncertainty matters here:** most anemia-screening demos report one confidence number and stop. When Monte Carlo Dropout's standard deviation across 30 passes exceeds our calibrated threshold, HemaVision explicitly tells the user the result is unreliable and recommends a repeat test — regardless of what the raw risk tier says. In live testing, a full-face phone photo (outside the model's training distribution of cropped conjunctiva images) correctly triggered this high-uncertainty warning.

---

## AMD Compute Usage

The reasoning layer runs on **gpt-oss-20b served via Fireworks AI's managed inference API**, which executes on **AMD Instinct MI300X GPUs**. This satisfies the hackathon's AMD compute requirement through the managed-API path listed alongside AMD Developer Cloud in the official hackathon description and AMD AI Developer Program member perks.

---

## Tech Stack

| Component | Technology |
|---|---|
| Image classifier | MobileNetV2 (transfer learning), TensorFlow/Keras |
| Uncertainty estimation | Monte Carlo Dropout (30 passes) |
| Reasoning layer | gpt-oss-20b via Fireworks AI (AMD MI300X) |
| Demo frontend | Gradio, deployed on Hugging Face Spaces |

---

## Repository Structure

```
HemaVision-Anemia-Triage/
├── README.md
├── app.py                        ← Gradio app (deployed on HF Spaces)
├── requirements.txt
├── LICENSE
├── Notebooks/
│   └── HemaVision.ipynb    ← End-to-end development & testing notebook
├── models/
│   └── mobilenet_final.keras     ← Trained MobileNetV2 classifier
└── docs/
    ├── Demo_Video.mp4            ← Full pitch + live demo walkthrough
    ├── Pitch_Deck.pptx           ← Presentation slides
   
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- TensorFlow 2.x
- A Fireworks AI API key ([app.fireworks.ai](https://app.fireworks.ai))

### Installation

```bash
git clone https://github.com/AminahAsif/HemaVision-Anemia-Triage.git
cd HemaVision-Anemia-Triage
pip install -r requirements.txt
```

### Run the demo locally

```bash
export FIREWORKS_API_KEY="your_key_here"
export MODEL_PATH="models/mobilenet_final.keras"
python app.py
```

### Reproduce the pipeline

Open [`Notebooks/HemaVision_Clean.ipynb`](Notebooks/HemaVision.ipynb) in Google Colab. Cell 2 (Configuration) is the only place you need to edit, set your model path, test image path, and API keys there, then run all cells top to bottom.

Research dataset: [Eyes-Defy-Anemia](https://ieee-dataport.org/documents/eyes-defy-anemia) (Dimauro et al., IEEE DataPort, DOI: 10.21227/t5s2-4j73)

---


## Honest Limitations

- This is a **screening aid, not a diagnostic device**  it does not replace laboratory hemoglobin testing
- The uncertainty threshold (0.095) is calibrated on a small validation sample and needs a larger dataset before real-world deployment
- The classifier is trained on **cropped conjunctiva images**; full-face photos require an automated cropping step not yet built
- Cross-validated model accuracy (62.5–74.5%) leaves real room for error , which is the reason the uncertainty layer exists in the first place

---

##  Clinical Disclaimer

This system is a research prototype and screening support tool developed for the AMD Developer Hackathon: ACT II. It is **not validated for clinical use** and must **not** be used to inform, replace, or influence any medical diagnosis or treatment decision. Always consult a qualified healthcare professional and laboratory testing for an actual diagnosis.

---

 
