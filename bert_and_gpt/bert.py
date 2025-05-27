import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# 加载数据
df = pd.read_csv("ChnSentiCorp_htl_all.csv")  # 确保路径正确
df = df[['review', 'label']].dropna()
df.columns = ['text', 'label']

# 划分训练和验证集
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 分词
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 加载模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 训练配置
training_args = TrainingArguments(
    output_dir="./bert_chinese_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 开始训练
trainer.train()

# 预测函数
def predict_sentiment(texts):
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return ["正面" if label == 1 else "负面" for label in predictions]

# 测试输入句子
examples = [
    "演员表演浮夸，完全无法让人产生代入感。",
    "食物份量十足，性价比超高，吃得很满足！"
]
results = predict_sentiment(examples)
for text, label in zip(examples, results):
    print(f"【{text}】 → 预测情感：{label}")
