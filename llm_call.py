from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("KKB_API")
client = OpenAI(base_url="https://mia.csp.kloudeks.com/v1", api_key=api_key)

# Sohbet tamamlama
resp = client.chat.completions.create(
   model="gpt-oss-120b",
   messages=[{"role": "user", "content": "Merhaba!"}],
)
print(resp.choices[0].message.content)

#