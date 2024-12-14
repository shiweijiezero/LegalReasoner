import argparse
import json
import os
import random
import re

from tqdm import tqdm
from transformers import AutoTokenizer
from jinja2 import Template, Environment, FileSystemLoader
import torch
from pprint import pprint
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)


def generate_prompt(content, tokenizer, jinja2_template, mode="claim"):
    """
    Generate a prompt for the document content.
    """
    if mode == "claim":
        prompt = jinja2_template.render(plaintiff_claim=content["plaintiff_claim"],
                                        facts=content["more_facts"])
    elif mode == "issue":
        prompt = jinja2_template.render(issue=content["issues"], plaintiff_claim=content["plaintiff_claim"],
                                        facts=content["more_facts"])
    elif mode == "generate_issues":
        prompt = jinja2_template.render(
            issue=content["generate_issues"],
            # issue=content["issues"],
            plaintiff_claim=content["plaintiff_claim"],
            facts=content["more_facts"]
        )
    else:
        raise ValueError("mode should be claim or issue")
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a Hong Kong legal assistant."},
            {"role": "user", "content": prompt}
        ],
        tokenize=False,  # 如果设为True,返回token ids;设为False返回文本
        add_generation_prompt=True  # 是否在最后添加生成提示符
    )
    return prompt


def eval_judge(res_contents, all_data, mode="claim"):
    """
    Calculate evaluation metrics for support/reject classification.

    Args:
        res_contents (list): List of model output strings
        all_data (list): List of ground truth data dictionaries

    Returns:
        dict: Dictionary containing evaluation metrics (accuracy, precision, recall, f1)
    """
    true_positives = 0  # 正确预测support
    false_positives = 0  # 错误预测support
    true_negatives = 0  # 正确预测reject
    false_negatives = 0  # 错误预测reject
    total = len(res_contents)

    for output, content in zip(res_contents, all_data):
        # 获取真实标签
        true_label = content["support&reject"].lower()
        # 获取预测标签
        if mode == "claim":
            if '"support"' in output.lower():
                pred_label = "support"
            elif '"reject"' in output.lower():
                pred_label = "reject"
            else:
                pred_label = "unknown"
        elif mode == "issue":
            # " # support # "
            pattern = re.compile(r'"\s*#\s*(\w+)\s*#\s*"')
            try:
                pred_label = pattern.findall(output)[0].lower()
            except:
                pred_label = "unknown"
        else:
            raise ValueError("mode should be claim or issue")
        # 统计各类样本数量
        if pred_label == "support":
            if true_label == "support":
                true_positives += 1
            else:
                false_positives += 1
        else:  # pred_label == "reject"
            if true_label == "reject":
                true_negatives += 1
            else:
                false_negatives += 1
        unknown_num = 0
        if pred_label == "unknown":
            unknown_num += 1

    # 计算评估指标
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0

    precision = (true_positives / (true_positives + false_positives)
                 if (true_positives + false_positives) > 0 else 0)

    recall = (true_positives / (true_positives + false_negatives)
              if (true_positives + false_negatives) > 0 else 0)

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    # 整理评估结果
    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "support_count": true_positives + false_negatives,
        "reject_count": true_negatives + false_positives,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "unknown_num": unknown_num,
        "total": total
    }

    # 打印评估结果
    pprint(metrics)

    return metrics


def main(args, model, proportion=1):
    base_path = args.base_path
    input_output_pairs = []

    # Eval by VLLM judge support or reject
    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=8192,
    )
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

    all_data = all_data[int(len(all_data) * (1 - proportion)):]
    logging.info(f"number of all_data: {len(all_data)}")
    # Start to judge
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载模板
    env = Environment(
        loader=FileSystemLoader('template'),  # templates是存放模板文件的目录
        trim_blocks=True,  # 删除块级标签后的第一个换行符
        lstrip_blocks=True  # 删除块级标签前的空白字符
    )
    # 加载具体模板文件
    jinja2_template = env.get_template('eval.jinja2')
    prompts = [generate_prompt(data, tokenizer, jinja2_template) for data in all_data]
    results = model.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in results]
    pprint(res_contents[0])
    # 抽取判决结果
    eval_res = eval_judge(res_contents, all_data)
    # Save results
    os.makedirs("eval_res", exist_ok=True)
    # 如果存在eval_res.json文件，读取并更新
    if os.path.exists("eval_res/eval_res.json"):
        with open("eval_res/eval_res.json", "r", encoding="utf-8") as f:
            eval_res_old = json.load(f)
    else:
        eval_res_old = []
    eval_res["model_name"] = args.model_name
    eval_res_old.append(eval_res)
    with open("eval_res/eval_res.json", "w", encoding="utf-8") as f:
        json.dump(eval_res_old, f, ensure_ascii=False, indent=4)


def main_issue(args, model, proportion=0.2):
    base_path = args.base_path
    input_output_pairs = []

    # Eval by VLLM judge support or reject
    # 设置生成参数
    sampling_params = SamplingParams(
        # temperature=0.7,
        # top_p=0.8,
        max_tokens=8192,
    )
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

    # random.seed(42)
    # random.shuffle(all_data)
    all_data = all_data[int(len(all_data) * (1 - proportion)):]
    logging.info(f"number of all_data: {len(all_data)}")
    # Start to judge
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载模板
    env = Environment(
        loader=FileSystemLoader('template'),  # templates是存放模板文件的目录
        trim_blocks=True,  # 删除块级标签后的第一个换行符
        lstrip_blocks=True  # 删除块级标签前的空白字符
    )
    # 加载具体模板文件
    jinja2_template = env.get_template('eval_issue.jinja2')

    if args.mode == "sc":
        repeat_num = 5
        prompts = [generate_prompt(data, tokenizer, jinja2_template, mode="generate_issues") for data in all_data]
        prompts = [prompt for prompt in prompts for _ in range(repeat_num)]
        results = model.generate(prompts, sampling_params=sampling_params)
        res_contents = [output.outputs[0].text for output in results]
        # Save intermediate results
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/sc_res.json", "w", encoding="utf-8") as f:
            json.dump(res_contents, f, ensure_ascii=False, indent=4)
        voted_res = []
        # 投票选出最终结果
        for i in range(0, len(res_contents), repeat_num):
            candidates = []
            for j in range(repeat_num):
                candidates.append(res_contents[i + j])
            # 抽取每个candidates的判决结果
            pred_labels = []
            for candidate in candidates:
                if '"support"' in candidate.lower():
                    pred_label = "support"
                elif '"reject"' in candidate.lower():
                    pred_label = "reject"
                else:
                    pred_label = "unknown"
                pred_labels.append(pred_label)
            # 投票
            support_count = pred_labels.count("support")
            reject_count = pred_labels.count("reject")
            unkonwn_count = pred_labels.count("unknown")
            # 三者最大的为最终结果
            if support_count >= reject_count and support_count >= unkonwn_count:
                final_res = '"support"'
            elif reject_count >= support_count and reject_count >= unkonwn_count:
                final_res = '"reject"'
            else:
                final_res = '"unknown"'
            voted_res.append(final_res)
        # 抽取判决结果
        eval_res = eval_judge(voted_res, all_data, mode="claim")
        eval_res["mode"] = "sc"
    elif args.mode == "prm":
        repeat_num = 5
        prompts = [generate_prompt(data, tokenizer, jinja2_template, mode="generate_issues") for data in all_data]
        prompts = [prompt for prompt in prompts for _ in range(repeat_num)]
        results = model.generate(prompts, sampling_params=sampling_params)
        res_contents = [output.outputs[0].text for output in results]
        # Save intermediate results
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/prm_res.json", "w", encoding="utf-8") as f:
            json.dump(res_contents, f, ensure_ascii=False, indent=4)
        exit()
    else:
        prompts = [generate_prompt(data, tokenizer, jinja2_template, mode="issue") for data in all_data]
        results = model.generate(prompts, sampling_params=sampling_params)
        res_contents = [output.outputs[0].text for output in results]
        pprint(res_contents[0])
        # 抽取判决结果
        eval_res = eval_judge(res_contents, all_data, mode="claim")
        eval_res["mode"] = "normal"
    # Save results
    os.makedirs("eval_res", exist_ok=True)
    # 如果存在eval_res.json文件，读取并更新
    if os.path.exists("eval_res/eval_res_issue.json"):
        with open("eval_res/eval_res_issue.json", "r", encoding="utf-8") as f:
            eval_res_old = json.load(f)
    else:
        eval_res_old = []
    eval_res["model_name"] = args.model_name
    eval_res_old.append(eval_res)
    with open("eval_res/eval_res_issue.json", "w", encoding="utf-8") as f:
        json.dump(eval_res_old, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--base_path", type=str, default="generate_issues_data_sft_issue_generate_qwen_7b")
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--gpu_count_num", type=int, default=1)
    parser.add_argument("--mode", type=str, default="normal")
    args = parser.parse_args()
    model = LLM(
        model=args.model_name,  # 模型路径
        trust_remote_code=True,  # Qwen 需要此参数
        tensor_parallel_size=args.gpu_count_num if args.gpu else torch.cuda.device_count(),  # 使用所有GPU
    )
    # main(args, model)
    main_issue(args, model)

    # model_name = "/home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/"
