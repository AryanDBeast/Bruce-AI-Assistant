# Bruce — AI Voice Assistant

Bruce is a voice-controlled desktop assistant written in Python, built in 2024 as a personal "Jarvis"-style project. It combines speech recognition, a large language model, text-to-speech, and several computer-vision models so you can talk to it and have it see through your webcam.

## How it works

1. **Face-ID login.** On startup Bruce loads reference photos from `known_faces/<name>/`, snaps a webcam frame, and uses `face_recognition` to confirm the authorised user before it does anything else.
2. **Wake word.** Bruce listens continuously through the microphone (Google Speech Recognition) and only acts on phrases that contain **"Bruce"**.
3. **Commands or conversation.** Recognised phrases trigger a built-in skill; anything else is sent to OpenAI (`gpt-4o-mini`) and the answer is spoken back.
4. **Speech out.** Replies are synthesised with Google Cloud Text-to-Speech (WaveNet voice) and played with `pygame`. For long answers, a background thread listens for "stop" / "shut up" so you can interrupt.

## Voice commands

| Say… | What Bruce does |
| --- | --- |
| "Bruce, take a picture" / "capture image" | Captures and saves a webcam photo |
| "Bruce, detect faces" | Counts and boxes faces with an OpenCV Haar cascade |
| "Bruce, what is this?" / "analyze this" | Classifies the main object with MobileNetV2 (ImageNet) |
| "Bruce, detect objects" / "analyze everything" | Detects multiple objects with Faster R-CNN (COCO) and draws boxes |
| "Bruce, read this" / "what does this say?" | Reads text aloud using Tesseract OCR |
| "Bruce, trivia time" | Asks a random trivia question and checks your spoken answer |
| "Bruce, <anything else>" | Answers with OpenAI and speaks the reply |

## Files

- `bruce_code.py`: the main assistant, covering the listen/think/speak loop and all vision skills.
- `Mics.py`: a small helper that lists microphone device indices so you can set `usable_mic` in `bruce_code.py`.
- `requirements.txt`: Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

You also need:

- **An OpenAI API key** in the environment: `export OPENAI_API_KEY=...`
- **Google Cloud Text-to-Speech credentials**: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`
- **Tesseract OCR** installed. The script points to the Windows default path (`C:\Program Files\Tesseract-OCR\tesseract.exe`), so change it for macOS or Linux.
- **Reference face photos** in `known_faces/<YourName>/`. Face photos are not included in this repo.
- The right microphone index. Run `python Mics.py` and set `usable_mic` in `bruce_code.py` to match.

Then run:

```bash
python bruce_code.py
```

## Tech stack

Python · OpenAI API · Google Cloud TTS · SpeechRecognition · OpenCV · TensorFlow/Keras (MobileNetV2) · PyTorch/torchvision (Faster R-CNN) · face_recognition · Tesseract · pygame
