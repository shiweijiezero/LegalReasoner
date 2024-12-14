import httpx
import asyncio
import json

from pprint import pprint

headers = {
    'Authorization': 'sk-Hd7x00Yozq5q7qvXC1BeA0E1853e4301Ac7450D5Ed69B003',
    'Content-Type': 'application/json',
}
# 设置并发数
CONCURRENCY = 200

# 创建一个信号量，用于控制并发数量
semaphore = asyncio.Semaphore(CONCURRENCY)


async def get_prompt(query):
    return f"""
# Task
"""

async def get_response(query):
    async with semaphore:
        async with httpx.AsyncClient(timeout=httpx.Timeout(360.0)) as client:
            prompt = await get_prompt(query)
            json_data = {
                'model': 'gpt-4',
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                'stream': False,
            }
            response = await client.post('https://api5.xhub.chat/v1/chat/completions', headers=headers, json=json_data)
            try:
                result = response.json()
                ans = result['choices'][0]['message']['content']
                res = json.loads(ans)
                return res
            except:
                return response.text
async def main(queries):
    print(f"Running {len(queries)} queries")
    tasks = [get_response(query) for query in queries]
    responses = await asyncio.gather(*tasks)
    return responses

import json
queries = ["你好"]

# 运行异步函数
responses = asyncio.run(main(queries))

results = []
for query, response in zip(queries, responses):
    results.append({"raw_material": query, "dialogue": response})
pprint(results)

# 保存对话数据
# output_file = 'dialogues.json'
# with open(output_file, 'w') as f:
#     json.dump(results, f, indent=2)


