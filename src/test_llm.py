from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.environ["LLM_API_KEY"],base_url=os.environ["LLM_BASE_URL"])

messages = [{"role":"user","content":"你好"}]
res = client.chat.completions.create(messages=messages,model=os.environ["LLM_MODEL"])
print(res.choices[0].message.content)