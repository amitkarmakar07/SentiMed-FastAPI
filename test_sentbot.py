import requests
import json

url = "http://localhost:8000/api/sentbot-ask"
payload = {
    "question": "Are the doctors good?",
    "hospital_names": None
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    with open("error_log.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Response saved to error_log.txt")
except Exception as e:
    print(f"Error: {e}")
