import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "glm-4.7-flash:latest",
        "prompt": "Respond with only UPWARD",
        "stream": False
    }
)

print(response.json()["response"])