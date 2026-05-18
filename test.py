# 测试天气 API
import requests
response = requests.get("https://wttr.in/Beijing?format=j1")
print("天气API状态:", response.status_code)

# 测试 Tavily API
from tavily import TavilyClient
tavily = TavilyClient(api_key="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")
try:
    result = tavily.search("test", search_depth="basic")
    print("Tavily API 连接成功")
except Exception as e:
    print("Tavily API 错误:", e)

# 测试 LLM API - ModelScope（如果您使用的是 ModelScope，请取消注释并替换配置）
from openai import OpenAI
client = OpenAI(
    api_key="ms-e02cd55c-2925-4a0d-9140-ff9511d654e2",
    base_url="https://api-inference.modelscope.cn/v1/"
)
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("LLM API 连接成功:", response.choices[0].message.content)
except Exception as e:
    print("LLM API 错误:", e)