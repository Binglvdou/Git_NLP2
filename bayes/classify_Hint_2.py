import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report  # 新增：导入分类评估报告函数
import warnings

# 忽略 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words

all_words = []

def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

# 设置高频词数量
top_num = 100

# 获取高频词
top_words = get_top_words(top_num)

# 构建特征矩阵
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
vector = np.array(vector)

# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1] * 127 + [0] * 24)

# 样本平衡处理
smote = SMOTE(random_state=42)  # 初始化 SMOTE
vector_balanced, labels_balanced = smote.fit_resample(vector, labels)  # 对特征和标签进行过采样
print("样本平衡后的类别分布：", Counter(labels_balanced))  # 打印平衡后的类别分布

# 训练模型
model = MultinomialNB()
model.fit(vector_balanced, labels_balanced)  # 使用平衡后的数据训练模型

def predict(filename):
    """对未知邮件分类"""
    # 构建未知邮件的词向量
    words = get_words(filename)
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'

# 预测并输出结果
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))

# 新增：模型评估
# 在训练集上预测
y_pred = model.predict(vector_balanced)
# 输出分类评估报告
print("\n模型评估报告：")
print(classification_report(labels_balanced, y_pred, target_names=["普通邮件", "垃圾邮件"]))