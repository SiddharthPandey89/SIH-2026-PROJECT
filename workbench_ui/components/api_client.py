import requests

BACKEND_URL = "http://localhost:8000"

def send_chat_message(prompt, session_id="default"):
    """POST /chat endpoint ko message bhejta hai."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat", 
            json={"prompt": prompt, "session_id": session_id},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {
            "response": "Backend server is currently offline.",
            "model_used": "Mock Local Engine",
            "error": str(e)
        }

def check_system_health():
    """GET /health endpoint check karta hai."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.json()
    except Exception:
        return {"status": "offline"}