import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return model, tokenizer

# 生成回复
def generate_response(model, tokenizer, input_text, max_length=100):
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

# 主程序
def main():
    # 加载训练好的模型和分词器
    model_dir = '/root/autodl-tmp/Llama-3.2-1B'  # 替换为你的模型路径
    model, tokenizer = load_model_and_tokenizer(model_dir)
    
    print("模型加载完成！输入 'exit' 退出。")
    while True:
        # 获取用户输入
        input_text = input("\n你: ")
        if input_text.lower() == 'exit':
            print("退出程序。")
            break
        
        # 生成回复
        response = generate_response(model, tokenizer, input_text)
        print(f"模型: {response}")

if __name__ == '__main__':
    main()
