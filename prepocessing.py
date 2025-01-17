import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig

# 设置环境变量以优化内存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. 读取和预处理对话数据
def preprocess_conversations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    conversations = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            input_text = lines[i].strip()
            output_text = lines[i + 1].strip()
            conversations.append((input_text, output_text))
    
    return conversations

# 2. 加载AutoTokenizer
def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# 3. Tokenize对话数据
def tokenize_conversations(conversations, tokenizer):
    inputs = []
    labels = []
    
    for input_text, output_text in conversations:
        input_encoding = tokenizer(input_text, truncation=True, padding='max_length', max_length=256)
        output_encoding = tokenizer(output_text, truncation=True, padding='max_length', max_length=256)
        
        inputs.append(input_encoding['input_ids'])
        labels.append(output_encoding['input_ids'])
    
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    
    return {'input_ids': inputs, 'labels': labels}

# 4. 创建训练数据集并分割为训练集和验证集
def create_dataset(conversations, tokenizer):
    tokenized_data = tokenize_conversations(conversations, tokenizer)
    dataset = Dataset.from_dict({
        'input_ids': tokenized_data['input_ids'],
        'labels': tokenized_data['labels'],
    })
    
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    return train_dataset, eval_dataset

# 5. 微调模型（使用LoRA）
def fine_tune_model(model, train_data, eval_data, output_dir='./results'):
    lora_config = LoraConfig(
        r=4,  # 降低LoRA的秩以节省内存
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # 减少训练轮次
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # 增加梯度累积步数
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=500,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )
    
    trainer.train()

# 6. 保存模型
def save_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# 7. 主程序入口
def main(file_path, hf_model_path):
    conversations = preprocess_conversations(file_path)
    tokenizer = get_tokenizer(hf_model_path)
    train_data, eval_data = create_dataset(conversations, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    fine_tune_model(model, train_data, eval_data)
    save_model(model, tokenizer, output_dir='./fine_tuned_llama_model')

if __name__ == '__main__':
    file_path = '/home/hui/Documents/bot/sexting-dataset/clean/conv1.txt'
    hf_model_path = './Llama-3.2-1B'
    main(file_path, hf_model_path)