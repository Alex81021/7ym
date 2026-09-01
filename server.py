import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import onnxruntime as ort
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="VoiceGuard In-Browser Engine")

MODEL_PATH = "deepfake_detector.onnx"
session = None
input_name = None
output_name = None

class FallbackDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

fallback_model = FallbackDetector()
fallback_model.eval()

try:
    if os.path.exists(MODEL_PATH):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"[Engine] Loaded ONNX model successfully.")
    else:
        print("[Engine Warning] ONNX file missing. Running PyTorch fallback model.")
except Exception as e:
    print(f"[Engine Warning] ONNX session init failed ({e}). Running PyTorch fallback.")

lfcc_transform = T.LFCC(sample_rate=16000, n_lfcc=60, speckwargs={"n_fft": 512, "hop_length": 160, "center": False})

def extract_lfcc(audio_window: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(audio_window.astype(np.float32)).unsqueeze(0)
    lfcc = lfcc_transform(waveform)
    lfcc = (lfcc - lfcc.mean(dim=-1, keepdim=True)) / (lfcc.std(dim=-1, keepdim=True) + 1e-6)
    return lfcc.unsqueeze(0)

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    if os.path.exists("dashboard.html"):
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>dashboard.html missing</h1>"

@app.websocket("/ws/browser")
async def browser_audio_stream(websocket: WebSocket):
    await websocket.accept()
    print("[Browser Engine] Client connected to live mic stream.")
    
    audio_buffer = np.array([], dtype=np.float32)
    REQUIRED_SAMPLES = 8000  # 500ms @ 16kHz

    try:
        while True:
            # Receive raw PCM 16kHz binary bytes from browser microphone
            raw_bytes = await websocket.receive_bytes()
            chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.append(audio_buffer, chunk)

            while len(audio_buffer) >= REQUIRED_SAMPLES:
                window = audio_buffer[:REQUIRED_SAMPLES]
                audio_buffer = audio_buffer[1600:]  # 100ms stride

                lfcc_tensor = extract_lfcc(window)

                if session is not None:
                    pred = session.run([output_name], {input_name: lfcc_tensor.numpy()})[0]
                    fake_score = float(pred[0][0])
                else:
                    with torch.no_grad():
                        fake_score = float(fallback_model(lfcc_tensor)[0][0])

                is_fake = fake_score > 0.50
                payload = {
                    "fake_probability": round(fake_score, 4),
                    "is_fake": is_fake,
                    "status": "ALERT: AI CLONED VOICE DETECTED" if is_fake else "VERIFIED: HUMAN VOICE"
                }

                await websocket.send_json(payload)

    except WebSocketDisconnect:
        print("[Browser Engine] Mic stream disconnected.")
    except Exception as e:
        print(f"[Engine Exception]: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)