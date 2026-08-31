import os
import json
import base64
import audioop
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import onnxruntime as ort
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse
from typing import Set

app = FastAPI(title="VoiceGuard 2-Way Call Engine")

# --- Configuration & Model Session ---
MODEL_PATH = "deepfake_detector.onnx"
TARGET_RECEIVER_NUMBER = "+919958284373"  # Replace with your phone number (E.164 format)
RENDER_APP_HOST = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")

session = None
input_name = None
output_name = None

try:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[Engine] ONNX model loaded. Input: '{input_name}', Output: '{output_name}'")
except Exception as e:
    print(f"[Engine Warning] ONNX load failed: {e}. Falling back to PyTorch model.")

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

lfcc_transform = T.LFCC(sample_rate=16000, n_lfcc=60, speckwargs={"n_fft": 512, "hop_length": 160, "center": False})

def extract_lfcc(audio_window: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(audio_window.astype(np.float32)).unsqueeze(0)
    lfcc = lfcc_transform(waveform)
    lfcc = (lfcc - lfcc.mean(dim=-1, keepdim=True)) / (lfcc.std(dim=-1, keepdim=True) + 1e-6)
    return lfcc.unsqueeze(0)

# --- Active Dashboard WebSocket Broadcast Connections ---
dashboard_connections: Set[WebSocket] = set()

async def broadcast_to_dashboards(data: dict):
    disconnected = set()
    for ws in dashboard_connections:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.add(ws)
    dashboard_connections.difference_update(disconnected)

# --- HTTP & Telephony Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    if os.path.exists("dashboard.html"):
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>dashboard.html missing</h1>"

@app.websocket("/ws/dashboard")
async def dashboard_ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    dashboard_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        dashboard_connections.remove(websocket)

@app.post("/incoming-call")
async def twilio_incoming_call():
    """TwiML response: Starts asynchronous media stream of inbound audio leg while bridging to destination target."""
    ws_url = f"wss://{RENDER_APP_HOST}/ws/twilio"
    twiml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{ws_url}" track="inbound_track" />
    </Start>
    <Dial>{TARGET_RECEIVER_NUMBER}</Dial>
</Response>"""
    return Response(content=twiml_xml, media_type="application/xml")

@app.websocket("/ws/twilio")
async def twilio_audio_stream(websocket: WebSocket):
    await websocket.accept()
    print("[Telephony Bridge] Incoming caller audio stream connected.")
    
    audio_buffer = np.array([], dtype=np.float32)
    REQUIRED_SAMPLES = 8000  # 500ms @ 16kHz

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data['event'] == 'media':
                # Decode 8kHz mulaw audio payload from Twilio
                raw_mulaw = base64.b64decode(data['media']['payload'])
                # Convert 8kHz mulaw -> 16kHz PCM
                pcm_8k = audioop.ulaw2lin(raw_mulaw, 2)
                pcm_16k = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)[0]

                chunk = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
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

                    # Broadcast telemetry score live to Chapter 3 Dashboard
                    await broadcast_to_dashboards(payload)

            elif data['event'] == 'stop':
                print("[Telephony Bridge] Call stream stopped.")
                break

    except WebSocketDisconnect:
        print("[Telephony Bridge] WebSocket connection closed.")
    except Exception as e:
        print(f"[Telephony Exception]: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)