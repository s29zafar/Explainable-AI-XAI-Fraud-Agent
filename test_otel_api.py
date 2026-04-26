import requests

url = "http://localhost:8000/investigate"
payload = {"tx_id": "TXN_000001439015"}

print(f"Sending request to {url} with payload {payload}...")
try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
