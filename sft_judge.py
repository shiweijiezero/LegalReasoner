import json
import logging
import os
import random
import subprocess
import sys

import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm


def generate_prompt(content, jinja2_template, mode="claim"):
    """
    Generate a prompt for the document content.
    """
    if mode == "claim":
        prompt = jinja2_template.render(plaintiff_claim=content["plaintiff_claim"],
                                        facts=content["more_facts"])
        response = "\n".join(
            content["judgment_decision"]) + f'\nSo In conclusion, the final decision is"{content["support&reject"]}"'
    elif mode == "issue":
        prompt = jinja2_template.render(issue=content["issues"], plaintiff_claim=content["plaintiff_claim"],
                                        facts=content["more_facts"])
    else:
        raise ValueError("mode should be claim or issue")
    prompt_dic = {"conversations":[
        {"role": "system", "content": "You are a Hong Kong legal assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]}
    return prompt_dic


def prepare_data_for_llama_factory(args):
    base_path = args.base_path
    input_output_pairs = []

    for root, dirs, _ in os.walk(base_path):
        if 'EN' in dirs:
            input_dir = os.path.join(root, 'EN')
            relative_path = os.path.relpath(input_dir, base_path)
            output_dir = os.path.join('filter_processed_data', relative_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            input_output_pairs.append((input_dir, output_dir))

    all_data = []
    for input_dir, output_dir in tqdm(input_output_pairs, desc="Processing files", total=len(input_output_pairs)):
        for file_name in tqdm(os.listdir(input_dir), desc="Processing list", total=len(os.listdir(input_dir))):
            if file_name.endswith('.json'):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(file_name)[0]}.json"
                )
                with (open(input_path, 'r', encoding='utf-8') as f):
                    content = json.load(f)
                    all_data.append(content)
    proportion = 0.6
    all_data = all_data[:int(len(all_data) * proportion)]
    # 加载模板
    env = Environment(
        loader=FileSystemLoader('template'),  # templates是存放模板文件的目录
        trim_blocks=True,  # 删除块级标签后的第一个换行符
        lstrip_blocks=True  # 删除块级标签前的空白字符
    )
    # 加载具体模板文件
    jinja2_template = env.get_template('eval_issue.jinja2')
    prompts = [generate_prompt(data, jinja2_template, mode="claim") for data in all_data]
    # 保存数据
    if not os.path.exists("data/sft_data"):
        os.makedirs("data/sft_data")
    with open(f"data/sft_data/sft.json", 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)
    return f"sft_data/sft.json"


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


def train_by_llama_factory(init_model, dataset_alias, output_dir, wandb_run_name, per_device_train_batch_size):
    if "llama" in init_model.lower():
        template_str = "llama3"
    elif "qwen" in init_model.lower():
        template_str = "qwen"
    cmd = f"""WANDB_API_KEY=75a0279c0374c55997f3d7736e7602a395d62da7 MKL_THREADING_LAYER=GNU DISABLE_VERSION_CHECK=1 llamafactory-cli train --model_name_or_path {init_model} --stage sft --deepspeed examples/deepspeed/ds_z3_config.json --do_train true --finetuning_type full --dataset {dataset_alias} --template {template_str} --cutoff_len 12800 --overwrite_cache true --preprocessing_num_workers 32 --output_dir {output_dir} --logging_steps 20 --save_steps 5000 --plot_loss true --overwrite_output_dir true --per_device_train_batch_size {per_device_train_batch_size} --gradient_accumulation_steps 1 --learning_rate 1.0e-4 --num_train_epochs 2.0 --lr_scheduler_type cosine --warmup_ratio 0.05 --bf16 true --ddp_timeout 180000000 --report_to wandb --run_name {wandb_run_name}"""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def main(args):
    # Set up seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    sft_data_path = prepare_data_for_llama_factory(args)
    dataset_alias = "sft"
    update_dataset_info(sft_data_path, dataset_alias)
    model_prefix = args.model_prefix
    # Training by LLAMA Factory
    if os.path.exists(f"sft_output/{model_prefix}"):
        os.makedirs(f"sft_output/{model_prefix}", exist_ok=True)
    train_by_llama_factory(
        init_model=args.init_model,
        dataset_alias=dataset_alias,
        output_dir=f"sft_output/{model_prefix}",
        wandb_run_name=f"sft_{model_prefix}",
        per_device_train_batch_size=4,
    )
    # Eval
    import eval
    from vllm import LLM, SamplingParams
    args.model_name = f"sft_output/{model_prefix}"
    model = LLM(
        model=args.model_name,  # 模型路径
        trust_remote_code=True,  # Qwen 需要此参数
        tensor_parallel_size=args.gpu_count_num if args.gpu else torch.cuda.device_count(),  # 使用所有GPU
    )
    eval.main(args, model, proportion=0.4)


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
