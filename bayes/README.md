# 文本分类项目

本项目实现了一个基于多项式朴素贝叶斯分类器的文本分类系统，支持高频词特征和TF-IDF加权特征两种特征选择模式。通过参数化切换机制，用户可以根据需求灵活选择特征提取方式。

## 代码核心功能说明

### 算法基础
本项目采用**多项式朴素贝叶斯分类器**（Multinomial Naive Bayes），其核心思想是基于贝叶斯定理和条件概率的特征独立性假设。

- **贝叶斯定理**：在邮件分类中，贝叶斯定理用于计算给定邮件内容条件下邮件属于某一类别的概率。具体形式为：
  
  $$P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)}$$
  
  其中：
  - $P(y|x)$ 是邮件内容 $x$ 属于类别 $y$ 的后验概率。
  - $P(x|y)$ 是类别 $y$ 下邮件内容 $x$ 的似然概率。
  - $P(y)$ 是类别 $y$ 的先验概率。
  - $P(x)$ 是邮件内容 $x$ 的边际概率。

- **特征独立性假设**：多项式朴素贝叶斯假设邮件中的每个词（特征）是相互独立的，即：
  
  $$P(x|y) = \prod_{i=1}^{n} P(x_i|y)$$
  
  其中 $x_i$ 表示邮件中的第 $i$ 个词。

### 数据处理流程
1. **分词处理**：使用 `jieba` 库对文本内容进行分词，将文本切分为独立的词语。
2. **停用词过滤**：通过正则表达式过滤无效字符（如标点符号、数字等），并去除长度为1的词。
3. **文本清理**：去除无关字符，保留有效词语，确保数据质量。

### 特征构建过程
本项目支持两种特征提取方式：
1. **高频词特征选择**：
   - 统计所有文本中出现频率最高的词，构建词频向量。
   - 数学表达形式：对每个文本，统计高频词在文本中的出现次数，形成特征向量。
   - 实现方式：使用 `collections.Counter` 统计词频，选择前 `top_num` 个高频词。

2. **TF-IDF加权特征**：
   - 计算每个词的TF-IDF值，衡量词在文本中的重要性。
   - 数学表达形式：
        
        $$\text{TF-IDF}(t, d) = \mathrm{TF}(t, d) \times \mathrm{IDF}(t)$$
        
        其中：
        - $\mathrm{TF}(t, d)$ 是词 $t$ 在邮件 $d$ 中的词频。
        - $\mathrm{IDF}(t)$ 是词 $t$ 的逆文档频率，计算公式为：
          
          $$\mathrm{IDF}(t) = \log \frac{N}{1 + \mathrm{DF}(t)}$$
          
          其中 $N$ 是总邮件数， $\mathrm{DF}(t)$ 是包含词 $t$ 的邮件数。
          <!-- 这里$\mathrm{DF}(t)$在GitHub仓库一直无法正常显示，原因是“，”与“$”之间缺少一个空格。-->
          <!-- 本来中文的“，”后面就自带一个迷惑性空格，导致一直没发现这个问题。 -->
    - 实现方式：使用 `sklearn.feature_extraction.text.TfidfVectorizer` 计算TF-IDF值。

### 高频词/TF-IDF两种特征模式的切换方法
在代码中，通过设置 `feature_type` 参数选择特征提取方式：
- 设置为 `'freq'` 时，使用高频词特征。
- 设置为 `'tfidf'` 时，使用TF-IDF加权特征。

示例：
```python
feature_type = 'freq'  # 使用高频词特征
# feature_type = 'tfidf'  # 使用TF-IDF加权特征
```
### 代码运行截图
`classify.py`

<img src="images/3.25_1.png" weight="800" alt="classify.py截图">

`classsify_TF.py`

<img src="images/3.25_2.png" weight="800" alt="classify_TF.py截图">

## optional
### 样本平衡处理
`classify_Hint.py`

<img src="images/3.25_3.png" weight="800" alt="classify_Hint.py截图">

### 增加模型评估指标
`classify_Hint_2.py`

<img src="images/3.25_4.png" weight="800" alt="classify_Hint.py截图">

