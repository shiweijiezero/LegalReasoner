import json
import os
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from jinja2 import Template, Environment, FileSystemLoader
import torch

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
    base_path = "data"

    # model, tokenizer
    # 初始化模型
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    # model_name = "/home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/"
    model = LLM(
        model=model_name,  # 模型路径
        trust_remote_code=True,  # Qwen 需要此参数
        tensor_parallel_size= torch.cuda.device_count(),  # 使用所有GPU
    )

    # 设置生成参数
    sampling_params = SamplingParams(
        # temperature=0.7,
        # top_p=0.8,
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
    jinja2_template = env.get_template('extract_data.jinja2')

    for root, dirs, _ in os.walk(base_path):
        if 'EN' in dirs:
            input_dir = os.path.join(root, 'EN')
            relative_path = os.path.relpath(input_dir, base_path)
            output_dir = os.path.join('processed_data', relative_path)
            input_output_pairs.append((input_dir, output_dir))

    all_data = []
    for input_dir, output_dir in tqdm(input_output_pairs, desc="Processing files", total=len(input_output_pairs)):
        for file_name in tqdm(os.listdir(input_dir), desc="Processing list", total=len(os.listdir(input_dir))):
            if file_name.endswith('.txt'):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(file_name)[0]}.json"
                )
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(list(content.split())) > 50000:
                        print(f"Skipped {input_path} - content too long")
                        continue
                    prompt = generate_prompt(content, tokenizer, jinja2_template)
                    all_data.append({
                        'input_path': input_path,
                        'output_path': output_path,
                        'content': content,
                        'prompt': prompt
                    })

    # all_data = all_data[:10]
    # vllm inference
    prompts = [data['prompt'] for data in all_data]
    results = model.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in results]
    # 抽取Json
    res_contents = [parse_json(output) for output in res_contents]
    print("number of results: ", len(res_contents))
    valid_contents = [content for content in res_contents if content is not None]
    print("number of valid results: ", len(valid_contents))

    # save results
    for data, res_content in zip(all_data, res_contents):
        os.makedirs(os.path.dirname(data['output_path']), exist_ok=True)
        if res_content is not None:
            try:
                with open(data['output_path'], 'w', encoding='utf-8') as f:
                    json.dump(res_content, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving {data['output_path']}: {e}")
                continue