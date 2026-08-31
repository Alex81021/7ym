import asyncio
import numpy as np
import websockets
import json

async def send_audio_stream(websocket):
    print("Streaming 16kHz PCM audio bytes...")
    # Send initial 500ms burst to fill audio buffer immediately
    initial_burst = (np.random.uniform(-0.5, 0.5, 8000) * 32767).astype(np.int16).tobytes()
    await websocket.send(initial_burst)
    
    # Stream continuous 100ms audio chunks
    for _ in range(20):
        pcm_chunk = (np.random.uniform(-0.5, 0.5, 1600) * 32767).astype(np.int16).tobytes()
        await websocket.send(pcm_chunk)
        await asyncio.sleep(0.05)

async def receive_predictions(websocket):
    frame_idx = 1
    try:
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(f"[Frame {frame_idx:02d}] Prob: {data['fake_probability']:.4f} | {data['status']}")
            frame_idx += 1
    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    uri = "ws://localhost:8000/ws/audio-stream"
    print(f"Connecting to audio engine at {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected successfully!")
        # Run sender and receiver concurrently
        await asyncio.gather(
            send_audio_stream(websocket),
            receive_predictions(websocket)
        )

if __name__ == "__main__":
    asyncio.run(main())