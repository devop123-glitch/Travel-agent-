import requests
import os
from dotenv import load_dotenv

load_dotenv()


url = "http://api.aviationstack.com/v1/airports"

params = {
    "access_key": os.getenv("AVIATIONSTACK_API_KEY"),
    "limit": 1,
}

response = requests.get(url, params=params, timeout=20)

print("HTTP status:", response.status_code)
print(response.text)