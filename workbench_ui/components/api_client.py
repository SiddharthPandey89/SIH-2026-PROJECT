import requests

BACKEND_URL = "http://localhost:8000"

def send_chat_message(prompt, session_id="default", file_id=None):
    try:
        payload = {
            "message": prompt,
            "conversation_id": session_id
        }
        if file_id:
            payload["file_id"] = file_id

        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json=payload,
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {
            "response": "Backend server is currently offline.",
            "model_used": "Mock Local Engine",
            "error": str(e)
        }