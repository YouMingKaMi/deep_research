from tavily import TavilyClient
import os

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
resp = client.search("what is deepseek")

print(resp)