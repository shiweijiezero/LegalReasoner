# 环境准备
最新的vllm

# 数据处理

1. extract_data.py
2. refine_data.py
3. filter_data.py

18401条数据

# 简单评测
eval.py

```bash
# python eval.py --gpu --gpu_count_num 4 --model_name /home/hansirui/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35/
python eval.py --gpu --gpu_count_num 4 --model_name Qwen/Qwen2.5-7B-Instruct
python eval.py --gpu --gpu_count_num 4 --model_name Qwen/Qwen2.5-14B-Instruct
python eval.py --gpu --gpu_count_num 4 --model_name /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/
python eval.py --gpu --gpu_count_num 4 --model_name /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/
python eval.py --gpu --gpu_count_num 4 --model_name /home/hansirui/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842/

CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --gpu --gpu_count_num 4 --model_name /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --gpu --gpu_count_num 4 --model_name /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --gpu --gpu_count_num 4 --model_name /home/hansirui/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842/

```

# 简单微调
```bash
python sft_judge.py --gpu --gpu_count_num 4 --init_model Qwen/Qwen2.5-7B-Instruct --model_prefix sft_qwen_7b --gpu --gpu_count_num 4
python sft_judge.py --gpu --gpu_count_num 4 --init_model Qwen/Qwen2.5-14B-Instruct --model_prefix sft_qwen_14b --gpu --gpu_count_num 4
python sft_judge.py --gpu --gpu_count_num 4 --init_model /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ --model_prefix sft_llama_8b --gpu --gpu_count_num 4
python sft_judge.py --gpu --gpu_count_num 4 --init_model /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/ --model_prefix sft_llama_70b --gpu --gpu_count_num 4
```

# 带issue简单微调
```bash
python sft_judge_issue.py --gpu --gpu_count_num 4 --init_model Qwen/Qwen2.5-7B-Instruct --model_prefix sft_issue_qwen_7b --gpu --gpu_count_num 4
python sft_judge_issue.py --gpu --gpu_count_num 4 --init_model Qwen/Qwen2.5-14B-Instruct --model_prefix sft_issue_qwen_14b --gpu --gpu_count_num 4
python sft_judge_issue.py --gpu --gpu_count_num 4 --init_model /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ --model_prefix sft_issue_llama_8b --gpu --gpu_count_num 4
```

# 微调issue生成
```bash
python sft_issue_generate.py --gpu --gpu_count_num 4 --init_model Qwen/Qwen2.5-7B-Instruct --model_prefix sft_issue_generate_qwen_7b 
python sft_issue_generate.py --gpu --gpu_count_num 4 --init_model /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ --model_prefix sft_issue_generate_llama_8b
python sft_issue_generate.py --gpu --gpu_count_num 4 --init_model Qwen/Qwen2.5-14B-Instruct --model_prefix sft_issue_generate_qwen_14b
```

# 微调基于issue生成推理
```bash
python sft_reason_generate.py  --init_model Qwen/Qwen2.5-7B-Instruct --model_prefix sft_reason_generate_qwen_7b --gpu --gpu_count_num 4
python sft_reason_generate.py --gpu --gpu_count_num 4 --init_model /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ --model_prefix sft_reason_generate_llama_8b
python sft_reason_generate.py --gpu --gpu_count_num 4 --init_model Qwen/Qwen2.5-14B-Instruct --model_prefix sft_reason_generate_qwen_14b

CUDA_VISIBLE_DEVICES=0,1,2,3 python sft_reason_generate.py --init_model /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ --model_prefix sft_reason_generate_llama_8b --gpu --gpu_count_num 4
CUDA_VISIBLE_DEVICES=0,1,2,3 python sft_reason_generate.py --init_model Qwen/Qwen2.5-14B-Instruct --model_prefix sft_reason_generate_qwen_14b --gpu --gpu_count_num 4
```

# 对推理过程进行打标签
```bash
sleep 3h    # 暂停1小时
python tag_reason.py  --init_model Qwen/Qwen2.5-7B-Instruct --model_prefix sft_reason_generate_qwen_7b --gpu --gpu_count_num 4
python tag_reason.py  --init_model /home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ --model_prefix sft_reason_generate_llama_8b --gpu --gpu_count_num 4
python tag_reason.py --init_model Qwen/Qwen2.5-14B-Instruct --model_prefix sft_reason_generate_qwen_14b --gpu --gpu_count_num 4
```

# Eval Best of N
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
python eval_best_of_n.py --mode sc --gpu --gpu_count_num 4 --model_name sft_output/sft_reason_generate_qwen_7b --base_path generate_issues_data_sft_issue_generate_qwen_7b
python eval_best_of_n.py --mode normal --gpu --gpu_count_num 4 --model_name sft_output/sft_reason_generate_qwen_7b --base_path generate_issues_data_sft_issue_generate_qwen_7b
```

Eval Prm:
```bash
python eval_best_of_n.py --mode prm --gpu --gpu_count_num 4 --model_name sft_output/sft_reason_generate_qwen_7b --base_path generate_issues_data_sft_issue_generate_qwen_7b
python eval_prm.py --gpu --gpu_count_num 4 --model_name ./prm_output/mistral-7b/ --base_path generate_issues_data_sft_issue_generate_qwen_7b

```