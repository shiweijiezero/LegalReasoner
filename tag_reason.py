import gc
import json
import logging
import os
import random
import subprocess
import sys
from pprint import pprint

import datasets
import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def generate_prompt(content, jinja2_template, tokenizer):
    """
    Generate a prompt for the document content.
    """
    prompt = jinja2_template.render(plaintiff_claim=content["plaintiff_claim"],
                                    facts=content["more_facts"],
                                    issue=content["generate_issues"])
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a Hong Kong legal assistant."},
            {"role": "user", "content": prompt}
        ],
        tokenize=False,  # 如果设为True,返回token ids;设为False返回文本
        add_generation_prompt=True  # 是否在最后添加生成提示符
    )
    prompt = prompt
    return prompt


def generate_multi_reason_prompt(content, jinja2_template, tokenizer):
    """
    Generate a prompt for the document content.
    """
    prompt = jinja2_template.render(plaintiff_claim=content["plaintiff_claim"],
                                    facts=content["more_facts"],
                                    issue=content["generate_issues"])
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a Hong Kong legal assistant."},
            {"role": "user", "content": prompt}
        ],
        tokenize=False,  # 如果设为True,返回token ids;设为False返回文本
        add_generation_prompt=True  # 是否在最后添加生成提示符
    )
    return prompt


def label_multi_reason_prompt(content, jinja2_template, tokenizer, repeat_num=3):
    """
    Generate a prompt for the document content.
    """
    prompts = []
    for j in range(repeat_num):
        referenced_court_reasoning = "\n".join(content["court_reasoning"]) + "\n".join(
            content["judgment_decision"]) + f'\nSo In conclusion, the final decision is"{content["support&reject"]}"'
        referenced_court_reasoning = referenced_court_reasoning.replace("\n", " <PRM> ")
        referenced_court_reasoning += " <PRM>"
        prompt = jinja2_template.render(
            plaintiff_claim=content["plaintiff_claim"],
            facts=content["more_facts"],
            issue=content["generate_issues"],
            referenced_court_reasoning=referenced_court_reasoning,
            model_reasoning=content[f"generate_reason_{j + 1}"]
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a Hong Kong legal assistant."},
                {"role": "user", "content": prompt}
            ],
            tokenize=False,  # 如果设为True,返回token ids;设为False返回文本
            add_generation_prompt=True  # 是否在最后添加生成提示符
        )
        prompts.append(prompt)
    return prompts


def parse_tag(tag):
    import re
    # 抽取列表
    try:
        tag_list = re.findall(r'\[(.*?)\]', tag)
        tag_list = tag_list[0]
        tag_list = eval(tag_list)
        tag_list = list(tag_list)
        # 检查里面只有 "+" 或者 "-"
        for t in tag_list:
            if t not in ["+", "-"]:
                return []
        if type(tag_list) != list:
            return []
        # print(f"tag_list: {tag_list}")
    except:
        tag_list = []
    return tag_list


def prepare_data_for_OpenRLHF(args):
    with open(f"tag_reason_data_{args.model_prefix}/data.json", 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # 过滤数据
    prm_data = []
    repeat_num = 10
    env = Environment(
        loader=FileSystemLoader('template'),  # templates是存放模板文件的目录
        trim_blocks=True,  # 删除块级标签后的第一个换行符
        lstrip_blocks=True  # 删除块级标签前的空白字符
    )
    # 加载具体模板文件
    jinja2_template = env.get_template('eval_issue.jinja2')
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    for data in tqdm(all_data, desc="Processing data", total=len(all_data)):
        for j in range(repeat_num):
            tag_data = data[f"tag_reason_{j + 1}"]
            tag_data = parse_tag(tag_data)
            generate_reason = data[f"generate_reason_{j + 1}"]
            generate_reason_lst = generate_reason.split("\n")
            # 比较长度：tag_data, generate_reason_lst
            # pprint(f"tag_data: {tag_data}")
            # pprint(f"generate_reason_lst: {generate_reason_lst}")
            # print(f"len(tag_data): {len(tag_data)}")
            # print(f"len(generate_reason_lst): {len(generate_reason_lst)}")
            if len(tag_data) != len(generate_reason_lst):
                continue
            else:
                data["prm_input"] = data[f"generate_reason_{j + 1}"].replace("\n", " ки ")
                data["prm_input"] = data["prm_input"] + " ки"
                # 顺序阅读data[f"generate_reason_{j + 1}"]，逐个将<PRM>替换为tag_data
                text = data[f"generate_reason_{j + 1}"]
                parts = text.split("\n")
                prm_label = ""
                for i in range(len(parts) - 1):
                    prm_label += parts[i] + f" {tag_data[i]} "
                prm_label += parts[-1] + f" {tag_data[-1]}"
                data["prm_label"] = prm_label
                # prm_data.append(
                #     {
                #         "conversations": [
                #             {"role": "system", "content": "You are a Hong Kong legal assistant."},
                #             {"role": "user", "content": data["prm_input"]},
                #             {"role": "assistant", "content": data["prm_label"]}
                #         ]
                #     }
                # )
                # 补充Question Prompt
                question_part = generate_prompt(data, jinja2_template, tokenizer)
                # 最后检查"ки"和"标签"个数是否一致
                # ки的个数
                count_ki = data["prm_input"].count("ки")
                # 标签的个数
                count_tag = len(tag_data)
                if count_ki != count_tag:
                    print(f"count_ki: {count_ki}, count_tag: {count_tag}")
                    continue
                if count_ki <= 2:
                    print(f"count_ki: {count_ki}, count_tag: {count_tag}")
                    continue
                prm_data.append(
                    {
                        "input": question_part + data["prm_input"],
                        "label": question_part + data["prm_label"],
                        "value": tag_data,
                    }
                )

    # 保存数据到
    if not os.path.exists("data/prm_data"):
        os.makedirs("data/prm_data")
    print(f"size of prm_data: {len(prm_data)}")
    with open(f"data/prm_data/prm_data.json", 'w', encoding='utf-8') as f:
        json.dump(prm_data, f, indent=4, ensure_ascii=False)
    # 转为dataset
    data = datasets.load_dataset("json", data_files=f"data/prm_data/prm_data.json")
    # 推到huggingface hub
    # data.push_to_hub("test_prm_data")
    data.save_to_disk("./prm_data/test/")

    return f"data/prm_data/prm_data.json"


def update_dataset_info(json_path, dataset_alias):
    dataset_info_path = f"data/dataset_info.json"
    if not os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=4)
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    logging.info(f"Updating dataset info for {dataset_alias} with file path {json_path}")
    dataset_info[dataset_alias] = {
        "file_name": json_path,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        }
    }

    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)


def train_by_OpenRLHF():
    cmd = f"""MKL_THREADING_LAYER=GNU DISABLE_VERSION_CHECK=1 deepspeed --module openrlhf.cli.train_prm --save_path ./prm_output/mistral-7b --save_steps 3000   --logging_steps 10    --eval_steps 300    --train_batch_size 32    --micro_train_batch_size 2    --pretrain mistralai/Mistral-7B-v0.1    --bf16    --max_epochs 1    --max_len 12800    --zero_stage 3    --learning_rate 1e-6   --dataset ./data/prm_data/prm_data.json    --input_key input    --label_key value    --flash_attn    --gradient_checkpointing    --packing_samples    --wandb_group prm    --placeholder_token ки    --reward_tokens + - --use_wandb 75a0279c0374c55997f3d7736e7602a395d62da7"""

    # """WANDB_API_KEY=75a0279c0374c55997f3d7736e7602a395d62da7 MKL_THREADING_LAYER=GNU DISABLE_VERSION_CHECK=1 deepspeed --module openrlhf.cli.train_prm --save_path ./prm_output/mistral-7b-math --save_steps 3000   --logging_steps 10    --eval_steps 300    --train_batch_size 32    --micro_train_batch_size 2    --pretrain mistralai/Mistral-7B-v0.1    --bf16    --max_epochs 1    --max_len 12800    --zero_stage 3    --learning_rate 1e-6   --dataset zhuzilin/Math-Shepherd    --input_key input    --label_key value    --flash_attn     --packing_samples    --wandb_group prm    --placeholder_token ки    --reward_tokens + - --use_wandb 75a0279c0374c55997f3d7736e7602a395d62da7"""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                               encoding='utf-8',  # 明确指定编码为 UTF-8
                               errors='replace',)
    for line in process.stdout:
        print(line, end='')
    sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def generate_multi_reasons(args, model, tokenizer):
    prefix = args.model_prefix
    prefix = prefix.replace("reason", "issue")  # sft_reason_generate_qwen_7b --> sft_issue_generate_qwen_7b
    base_path = f"generate_issues_data_{prefix}"
    input_output_pairs = []

    for root, dirs, _ in os.walk(base_path):
        if 'EN' in dirs:
            input_dir = os.path.join(root, 'EN')
            relative_path = os.path.relpath(input_dir, base_path)
            output_dir = os.path.join(f'generate_reason_data_{args.model_prefix}', relative_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            input_output_pairs.append((input_dir, output_dir))

    all_data = []
    output_files = []
    for input_dir, output_dir in tqdm(input_output_pairs, desc="Processing files", total=len(input_output_pairs)):
        for file_name in tqdm(os.listdir(input_dir), desc="Processing list", total=len(os.listdir(input_dir))):
            if file_name.endswith('.json'):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(file_name)[0]}.json"
                )
                output_files.append(output_path)
                with (open(input_path, 'r', encoding='utf-8') as f):
                    content = json.load(f)
                    all_data.append(content)
    proportion = 0.01
    all_data = all_data[:int(len(all_data) * proportion)]
    # 加载模板
    env = Environment(
        loader=FileSystemLoader('template'),  # templates是存放模板文件的目录
        trim_blocks=True,  # 删除块级标签后的第一个换行符
        lstrip_blocks=True  # 删除块级标签前的空白字符
    )
    # 加载具体模板文件
    jinja2_template = env.get_template('eval_issue.jinja2')
    repeat_num = 10
    prompts = [generate_multi_reason_prompt(data, jinja2_template, tokenizer) for data in all_data]
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts.extend([prompt] * repeat_num)
    # 生成issues
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=8192,
    )
    results = model.generate(repeated_prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in results]
    # all_data[i][f"generate_reason_{j}"]
    for i in range(len(all_data)):
        for j in range(repeat_num):
            all_data[i][f"generate_reason_{j + 1}"] = res_contents[i * repeat_num + j]
    # 保存数据到output_files
    output_path_file = f"generate_reason_data_{args.model_prefix}/data.json"
    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    with open(output_path_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


def tag_reasons(tokenizer, model):
    base_path_file = f"generate_reason_data_{args.model_prefix}/data.json"
    with open(base_path_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    # all_data = all_data[:100]
    # 加载模板
    env = Environment(
        loader=FileSystemLoader('template'),  # templates是存放模板文件的目录
        trim_blocks=True,  # 删除块级标签后的第一个换行符
        lstrip_blocks=True  # 删除块级标签前的空白字符
    )
    # 加载具体模板文件
    jinja2_template = env.get_template('tag_reason.jinja2')
    repeat_num = 10
    prompts = []
    for data in all_data:
        tmp_prompts = label_multi_reason_prompt(data, jinja2_template, tokenizer, repeat_num)
        prompts.extend(tmp_prompts)
    # 生成issues
    sampling_params = SamplingParams(
        max_tokens=8192,
    )
    results = model.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in results]
    # all_data[i][f"tag_reason_{j}"]
    for i in range(len(all_data)):
        for j in range(repeat_num):
            all_data[i][f"tag_reason_{j + 1}"] = res_contents[i * repeat_num + j]
    # 保存数据到output_files
    output_path_file = f"tag_reason_data_{args.model_prefix}/data.json"
    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    with open(output_path_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


def main(args):
    # Set up seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model_prefix = args.model_prefix
    # Generate issues
    args.model_name = f"sft_output/{model_prefix}"
    # model = LLM(
    #     model=args.model_name,  # 模型路径
    #     trust_remote_code=True,  # Qwen 需要此参数
    #     tensor_parallel_size=args.gpu_count_num if args.gpu else torch.cuda.device_count(),  # 使用所有GPU
    #     # disable_async_output_proc=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # generate_multi_reasons(args, model, tokenizer)
    # # Release GPU memory
    # from vllm.distributed.parallel_state import destroy_model_parallel
    # destroy_model_parallel()
    # del model.llm_engine.model_executor.driver_worker
    # del model # Isn't necessary for releasing memory, but why not
    # gc.collect()
    # torch.cuda.empty_cache()

    # Tag reasons
    model = LLM(
        model="Qwen/Qwen2.5-14B-Instruct",  # 模型路径
        trust_remote_code=True,  # Qwen 需要此参数
        tensor_parallel_size=args.gpu_count_num if args.gpu else torch.cuda.device_count(),  # 使用所有GPU
        # disable_async_output_proc=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tag_reasons(tokenizer, model)
    # Release GPU memory
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    del model.llm_engine.model_executor.driver_worker
    del model # Isn't necessary for releasing memory, but why not
    gc.collect()
    torch.cuda.empty_cache()

    # Train PRM by OpenRLHF
    prepare_data_for_OpenRLHF(args)
    train_by_OpenRLHF()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="filter_processed_data")
    parser.add_argument("--init_model", type=str, default="")
    parser.add_argument("--model_prefix", type=str, default="sft")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--gpu_count_num", type=int, default=1)
    args = parser.parse_args()
    main(args)
