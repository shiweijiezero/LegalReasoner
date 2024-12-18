import json
import os
import re

import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from jinja2 import Template, Environment, FileSystemLoader
import torch
from ..clause_utils import build_hierarchy

def generate_prompt(content, tokenizer, jinja2_template):
    """
    Generate a prompt for the document content.
    """
    prompt = jinja2_template.render(content=content)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": prompt}
        ],
        tokenize=False,  # 如果设为True,返回token ids;设为False返回文本
        add_generation_prompt=True  # 是否在最后添加生成提示符
    )
    return prompt

def parse_json(output):
    # 使用正则表达式匹配最外层的大括号
    json_match = re.search(r'\{.*\}', output, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            # 尝试解析JSON字符串
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            # print(f"JSON解析错误: {e}")
            return None
    else:
        # print("未找到有效的JSON内容")
        return None


if __name__ == "__main__":
    input_output_pairs = []
    base_path = "spider"

    # model, tokenizer
    # 初始化模型
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = LLM(
        model=model_name,  # 模型路径
        trust_remote_code=True,  # Qwen 需要此参数
        tensor_parallel_size= torch.cuda.device_count(),  # 使用所有GPU
    )

    # 设置生成参数
    sampling_params = SamplingParams(
        max_tokens=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载模板
    env = Environment(
        loader=FileSystemLoader('template'),  # templates是存放模板文件的目录
        trim_blocks=True,  # 删除块级标签后的第一个换行符
        lstrip_blocks=True  # 删除块级标签前的空白字符
    )
    # 加载具体模板文件
    jinja2_template = env.get_template('label_chatlaw.jinja2')

    # 构建法条层级
    hierarchy_law = build_hierarchy(base_path)
    caps = hierarchy_law.get_caps()
    caps_list = [f"{cap['lower_cap_number']} {cap['lower_title']}: {cap['long_title']}" for key, cap in caps.items()]

    all_data = datasets.load_dataset("Aarushhh/law-questions-4k", split="train")
    all_data = all_data["prompts"].to_list()
    for question in tqdm(all_data, desc="Processing questions", total=len(all_data)):
        # 对caps_list每10个进行一个处理
        clause_chunks = [caps_list[i:i + 10] for i in range(0, len(caps_list), 10)]
        for clause_chunk in clause_chunks:
            prompt = generate_prompt(question, tokenizer, jinja2_template, clause_chunk)
            all_data.append({
                'prompt': prompt,
                'question': question,
                'clause_chunk': clause_chunk
            })

    # vllm inference
    prompts = [data['prompt'] for data in all_data]
    results = model.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in results]

    # save results
    for data, res_content in zip(all_data, res_contents):
        output_path = os.path.join("processed_data", f"{os.path.splitext(question_file)[0]}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'question': data['question'],
                'answer': res_content,
                'relevant_clauses': data['relevant_clauses']
            }, f, indent=4, ensure_ascii=False)