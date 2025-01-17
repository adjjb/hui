import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载训练好的模型和分词器
def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return model, tokenizer

# 2. 生成文本
def generate_text(model, tokenizer, input_text, max_length=100):
    # 将输入文本编码为模型输入
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # 将输入数据移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,  # 生成一个结果
            no_repeat_ngram_size=2,  # 避免重复的 n-gram
            top_k=50,  # 限制采样范围
            top_p=0.95,  # 使用 nucleus sampling
            temperature=0.7,  # 控制随机性
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# 3. 测试模型
def test_model(model_dir, test_inputs):
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    # 对每个测试输入生成文本
    for input_text in test_inputs:
        print(f"Input: {input_text}")
        generated_text = generate_text(model, tokenizer, input_text)
        print(f"Generated Output: {generated_text}")
        print("-" * 50)

# 4. 主程序入口
if __name__ == '__main__':
    # 训练好的模型目录
    model_dir = './fine_tuned_llama_model'
    
    # 测试输入
    test_inputs = [
       "Just We!"
    ]
    
    # 测试模型
    test_model(model_dir, test_inputs)
