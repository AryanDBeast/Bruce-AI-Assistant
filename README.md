# Bruce — AI Voice Assistant

Bruce is a voice-controlled desktop assistant written in Python. You talk to it, it talks back, and it can see through your webcam. It combines speech recognition, a large language model, neural text-to-speech, and four computer-vision models in a single program, with face recognition standing in for a password.

The voice pipeline built here (listen → LLM → speak) later became the starting point for [AI Caretaker](https://github.com/AryanDBeast/AI-Caretaker).

---

## How it works

```
 webcam ──▶ Face-ID check ──✗──▶ exit
                 │ ✓
                 ▼
 mic ──▶ Speech-to-text ──▶ contains "Bruce"? ──no──▶ ignore
                                   │ yes
                 ┌─────────────────┴──────────────────┐
                 ▼                                    ▼
        matches a vision skill?               anything else
     (photo, faces, objects, OCR, trivia)     → OpenAI chat
                 │                                    │
                 └──────────────▶ Google TTS ◀────────┘
                                  (interruptible)
```

1. **Face-ID login.** On startup Bruce encodes the reference photos in `known_faces/<name>/`, captures a webcam frame, and uses `face_recognition` to confirm the authorized user before doing anything else.
2. **Wake word.** It listens continuously (Google Speech Recognition) and only acts on phrases that contain **"Bruce"**.
3. **Skills or conversation.** Recognized phrases trigger a built-in skill. Anything else goes to OpenAI (`gpt-4o-mini`), and the answer is cleaned of Markdown and spoken back.
4. **Speech out.** Replies are synthesized with Google Cloud Text-to-Speech (WaveNet voice) and played with `pygame`. For long answers, a background thread listens for "stop" or "shut up" so you can interrupt.

## Voice commands

| Say… | What Bruce does | Model |
| --- | --- | --- |
| "Bruce, take a picture" | Captures and saves a webcam photo | — |
| "Bruce, detect faces" | Counts and boxes every face in frame | OpenCV Haar cascade |
| "Bruce, what is this?" | Names the main object in view | MobileNetV2 (ImageNet) |
| "Bruce, detect objects" | Detects and labels multiple objects | Faster R-CNN (COCO) |
| "Bruce, read this" | Reads printed text aloud | Tesseract OCR |
| "Bruce, trivia time" | Asks a question and checks your spoken answer | — |
| "Bruce, *anything else*" | Answers conversationally | OpenAI `gpt-4o-mini` |

---

## Setup

```bash
pip install -r requirements.txt
```

Then set up the following:

| Requirement | How |
| --- | --- |
| OpenAI API key | `export OPENAI_API_KEY=...` |
| Google Cloud TTS credentials | `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` |
| Tesseract OCR | Install it, then update `tesseract_cmd` in `bruce_code.py` if you're not on the Windows default path |
| Reference face photos | Add `.jpg`/`.png` photos to `known_faces/authorized_user/` (no photos are included in this repo) |
| Your name (optional) | `export BRUCE_USER="Sam"`. This is the name Bruce greets you by. |
| Microphone index | Run `python Mics.py`, then `export BRUCE_MIC_INDEX=<n>` |

Run it:

```bash
python bruce_code.py
```

## Files

| File | Purpose |
| --- | --- |
| `bruce_code.py` | The assistant: listen/think/speak loop, face-ID, and all vision skills |
| `Mics.py` | Lists microphone device indices |
| `requirements.txt` | Python dependencies |
| `known_faces/` | Reference photos for face-ID, one folder per person |

## Tech stack

Python · OpenAI API · Google Cloud Text-to-Speech · SpeechRecognition · OpenCV · TensorFlow/Keras (MobileNetV2) · PyTorch/torchvision (Faster R-CNN) · face_recognition · Tesseract · pygame

---

## Version history

Bruce was developed locally through 2024 and published here as a single cleaned-up commit, so its earlier stages are summarized below.

| Version | Date | What changed |
| --- | --- | --- |
| **v0.1** – Talking assistant | Summer 2024 | Microphone → speech-to-text → OpenAI → Google TTS loop. Conversation memory stored in a spreadsheet and fed back as context. |
| **v0.2** – Face-ID | Aug–Sep 2024 | Webcam login with `face_recognition`; Haar-cascade face detection and photo capture. Experimented with a custom-trained Keras image model. |
| **v0.3** – Vision skills | Fall 2024 | MobileNetV2 object classification, Faster R-CNN multi-object detection with bounding boxes, and Tesseract OCR. |
| **v1.0** – Full assistant | Dec 2024 | Final local version: "Bruce" wake word, voice trivia game, interruptible speech for long answers, Markdown stripped from replies before speaking. |
| **v1.0.1** – Public release | Sep 2026 | Published to GitHub. API keys moved to environment variables; large model files, captured images and face photos excluded via `.gitignore`. |
| **v1.1** – Configurable user | Oct 2026 | Authorized user name and mic index read from environment variables instead of being hardcoded. Face loader now skips non-image files, which fixes a crash on a fresh clone. |
