import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "glm-4.7-flash:latest"


def ask_ollama(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=105
    )

    response.raise_for_status()

    return response.json()["response"]