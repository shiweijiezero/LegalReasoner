# Legal Reasoner

## Overview
This repository contains code and tools for processing, analyzing, and fine-tuning language models on legal case data, with a specific focus on Hong Kong legal documents (LegalHK dataset).

## Environment Setup
- Requires latest version of vllm
- Additional dependencies can be found in `requirements.txt` (to be created)

## Data Processing Pipeline

### LegalHK Dataset Generation
1. Extract raw data:
```bash
python extract_data.py
```

2. Refine the extracted data:
```bash
python refine_data.py
```

3. Filter and clean the data:
```bash
python filter_data.py
```

The processed LegalHK dataset is available at https://huggingface.co/datasets/weijiezz/LegalHK.

## Model Evaluation

### Basic Evaluation
Run evaluations on various models using:
```bash
python eval.py --gpu --gpu_count_num 4 --model_name MODEL_PATH
```

Supported models include:
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Qwen2.5-72B-Instruct
- Llama-3.1-8B-Instruct
- Llama-3.1-70B-Instruct

For specific GPU allocation, use:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --gpu --gpu_count_num 4 --model_name MODEL_PATH
```

## Model Fine-tuning

### Basic Fine-tuning
```bash
python sft_judge.py --gpu --gpu_count_num 4 --init_model MODEL_PATH --model_prefix PREFIX
```

### Issue-aware Fine-tuning
```bash
python sft_judge_issue.py --gpu --gpu_count_num 4 --init_model MODEL_PATH --model_prefix PREFIX
```

### Issue Generation Fine-tuning
```bash
python sft_issue_generate.py --gpu --gpu_count_num 4 --init_model MODEL_PATH --model_prefix PREFIX
```

### Reasoning Generation Fine-tuning
```bash
python sft_reason_generate.py --gpu --gpu_count_num 4 --init_model MODEL_PATH --model_prefix PREFIX
```

## Tagging and Analysis

### Reasoning Process Tagging
```bash
python tag_reason.py --init_model MODEL_PATH --model_prefix PREFIX --gpu --gpu_count_num 4
```

### Process Verification
Three aspects are evaluated:

1. Correctness: Use tag_reason_correctness.jinja2 template
2. Progressiveness: Use tag_reason_progressiveness.jinja2 template
3. Potential: Evaluated using Best-of-N approach during eval_best_of_n.py execution

For classifier training, use tag_reason.py with the classifier.jinja2 template.

### Best-of-N Evaluation
```bash
python eval_best_of_n.py --mode [sc|normal|prm] --gpu --gpu_count_num 4 \
    --model_name MODEL_PATH \
    --base_path BASE_PATH
```

## Deployment Services

### Legal Clause Retrieval Service
```bash
uvicorn stationary_locate/trace_clause_gpt_fastapi:app --host 0.0.0.0 --port 8123
```

### Case Retrieval Service
Deploy using [DIFY](https://github.com/langgenius/dify) and upload case data to the platform.

## LLaMA Factory Integration
Template for fine-tuning with LLaMA Factory:
```bash
MKL_THREADING_LAYER=GNU DISABLE_VERSION_CHECK=1 llamafactory-cli train \
    --model_name_or_path {init_model} \
    --stage sft \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --do_train true \
    --finetuning_type full \
    --dataset {dataset_alias} \
    --template {template_str} \
    --cutoff_len 12800 \
    --overwrite_cache true \
    --preprocessing_num_workers 32 \
    --output_dir {output_dir} \
    --logging_steps 20 \
    --save_steps 5000 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size {per_device_train_batch_size} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --report_to wandb \
    --run_name {wandb_run_name}
```