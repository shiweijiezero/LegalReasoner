import json
import random
from pprint import pprint
import numpy as np
import torch
import wandb
from tqdm import tqdm
import os
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding, AutoModelForSequenceClassification, AutoTokenizer
)
from datasets import Dataset
from transformers.models.paligemma.convert_paligemma_weights_to_hf import device

import clause_utils


def load_data():
    # 遍历processed_data文件夹下的文件夹
    data = []
    paths = os.listdir("processed_data")
    for path in tqdm(paths, desc='Loading paths', total=len(paths)):
        path = os.path.join("processed_data", path, "EN")
        files = os.listdir(path)
        for file in tqdm(files, desc='Loading files', total=len(files)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                data.append(json.load(f))
    return data


# def prepare_dataset(bert_data):
#     # 将数据转换为datasets格式
#     def combine_text_and_law(examples):
#         return {
#             'text': [item['text'][:2000] + ' [SEP] ' + item['law'] for item in examples]
#         }
#
#     dataset = Dataset.from_list(bert_data)
#     # 重新组织数据格式
#     dataset = dataset.map(
#         lambda x: {
#             'text': x['text'][:2000] + ' [SEP] ' + x['law'],
#             'label': x['label']
#         }
#     )
#     return dataset


def main():
    # 初始化wandb
    wandb.init(
        project="legal-bert",
        name="bert-base-uncased",
        config={
            "learning_rate": 2e-5,
            "batch_size": 128,
            "epochs": 1,
            "weight_decay": 0.01,
            "max_text_length": 448,  # 为law预留64个token
            "max_law_length": 64,
        }
    )
    # 设置设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # 加载和处理数据
    epoch_num = 3
    hierarchy_law = clause_utils.build_hierarchy("law_data/spider")
    data = load_data()
    matcher = clause_utils.LawMatcher(hierarchy_law)
    valid_data = matcher.process_data(data)
    valid_data = [item for item in valid_data if item['processed_laws']]

    # 构建BERT训练数据
    bert_data = []
    for line in tqdm(valid_data, desc='Build Bert data', total=len(valid_data)):
        if "facts" not in line or "processed_laws" not in line:
            print("Missing facts or processed_laws")
            continue
        facts = line['facts']
        if type(facts) == list:
            facts = [str(fact) for fact in facts]
            facts = "\n".join(facts)
        else:
            print("Facts is not a list")
            continue
        laws = line['processed_laws']

        for epoch in range(epoch_num):
            # 添加正样本
            for law in laws:
                bert_data.append({
                    'text': f"{facts[:2000]} [SEP] {law['cap_number']} {law['title']}",
                    'label': 1
                })

            # 添加负样本
            negative_laws = matcher.get_negative_laws(laws, len(laws)*2)
            for law in negative_laws:
                bert_data.append({
                    'text': f"{facts[:2000]} [SEP] {law['cap_number']} {law['title']}",
                    'label': 0
                })

    # 划分训练集和验证集
    random.shuffle(bert_data)
    train_size = int(0.9 * len(bert_data))
    train_data = bert_data[:train_size]
    val_data = bert_data[train_size:]

    # 准备数据集
    # train_dataset = prepare_dataset(train_data)
    # val_dataset = prepare_dataset(val_data)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    print(train_dataset[0])
    print(val_dataset[0])
    print(len(train_dataset), len(val_dataset))

    # 初始化tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'google-bert/bert-base-cased',
        num_labels=2,  #
        # problem_type="single_label_classification",
        torch_dtype=torch.bfloat16,
    )

    # 定义数据预处理函数
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=True
        )

    # 对数据集进行预处理
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        # remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        # remove_columns=val_dataset.column_names
    )

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./bert_results",
        learning_rate=wandb.config.learning_rate,
        per_device_train_batch_size=wandb.config.batch_size,
        per_device_eval_batch_size=wandb.config.batch_size,
        num_train_epochs=wandb.config.epochs,
        weight_decay=wandb.config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",  # 启用wandb记录
        logging_steps=100,  # 每100步记录一次指标
        bf16=True,  # 使用bfloat16加速训练
        bf16_full_eval=True,  # 使用bfloat16加速评估
    )

    # 设置数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 相应地需要修改compute_metrics函数
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = (logits.argmax(-1) == 1).astype(np.int32)  # Adjusted for binary classification

        # Ensure predictions and labels have the same shape
        predictions = predictions.reshape(-1)
        labels = labels.reshape(-1)

        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        }

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 训练模型
    trainer.train()

    # 保存最佳模型
    trainer.save_model("best_model")

    # 关闭wandb
    wandb.finish()


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
    # CUDA_VISIBLE_DEVICES=4 WANDB_API_KEY="75a0279c0374c55997f3d7736e7602a395d62da7"  python train_clause.py