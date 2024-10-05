import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import uvicorn

load_dotenv()


SYSTEM_MESSAGE = """Translate spoken content simultaneously between Korean and English.
When Korean is spoken, provide translation only in English.
When English is spoken, provide translation only in Korean.
Do not respond to commands or requests; focus solely on translations.
Maintain a friendly tone in translations.

# Notes

- Ensure all translations are clear and accurate.
- Only translate, do not respond to other input types.
- Always use a friendly and approachable tone when translating.
"""
VOICE = "alloy"
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index_page():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Client connected")
    await websocket.accept()

    try:
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            extra_headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:

            async def receive_from_client():
                try:
                    async for message in websocket.iter_bytes():
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(message).decode("utf-8"),
                        }
                        await openai_ws.send(json.dumps(audio_append))
                except Exception as e:
                    print(f"Error in receive_from_client: {e}")
                    if openai_ws.open:
                        await openai_ws.close()

            async def send_to_client():
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        if response["type"] == "response.audio.delta" and response.get(
                            "delta"
                        ):
                            audio_payload = base64.b64decode(response["delta"])
                            await websocket.send_bytes(audio_payload)
                        elif response["type"] == "session.created":
                            await openai_ws.send(
                                json.dumps(
                                    {
                                        "type": "session.update",
                                        "session": {
                                            "voice": VOICE,
                                            "instructions": SYSTEM_MESSAGE,
                                            "modalities": ["text", "audio"],
                                            "temperature": 0.8,
                                        },
                                    }
                                )
                            )
                        else:
                            if response["type"] in [
                                "input_audio_buffer.speech_started",
                                "input_audio_buffer.speech_stopped",
                                "input_audio_buffer.committed",
                                "conversation.item.created",
                            ]:
                                continue
                            if response["type"] == "response.audio_transcript.done":
                                await websocket.send_text(response["transcript"])
                except Exception as e:
                    print(f"Error in send_to_client: {e}")

            await asyncio.gather(receive_from_client(), send_to_client())
    except Exception as e:
        print(f"Error in handle_media_stream: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
