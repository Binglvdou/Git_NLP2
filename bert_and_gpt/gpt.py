from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 本地模型路径
local_model_path = "gpt2-chinese-cluecorpussmall"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)
model.eval()

# 设置输入句子
prompt = "我走进了那扇从未打开过的门"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# 生成文本（续写）
output = model.generate(
    input_ids,
    max_length=100,  # 输出总长度（包含原始prompt）
    do_sample=True,  # 启用采样（否则是贪婪搜索）
    top_k=50,  # 限定前 k 个概率最高的词
    top_p=0.95,  # nucleus sampling
    temperature=0.9,  # 控制生成多样性
    num_return_sequences=1  # 返回几条
)

# 解码结果
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
