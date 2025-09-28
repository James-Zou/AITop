# 数据预处理

## 1. 数据预处理概述

### 定义
数据预处理是将原始数据转换为适合机器学习算法使用的格式的过程。

### 重要性
- **数据质量**：提高数据质量，减少噪声
- **算法性能**：改善算法性能，提高准确率
- **计算效率**：减少计算复杂度，提高效率
- **模型稳定性**：提高模型稳定性，减少过拟合

### 预处理步骤
1. 数据收集
2. 数据清洗
3. 数据集成
4. 数据变换
5. 数据规约
6. 数据分割

## 2. 数据清洗

### 定义
数据清洗是识别和纠正数据中错误、不完整、不准确或不相关的部分的过程。

### 常见问题

**缺失值**:
- 完全随机缺失(MCAR)
- 随机缺失(MAR)
- 非随机缺失(MNAR)

**异常值**:
- 统计异常值
- 业务异常值
- 技术异常值

**重复数据**:
- 完全重复
- 部分重复
- 近似重复

**不一致数据**:
- 格式不一致
- 单位不一致
- 编码不一致

### 处理方法

**缺失值处理**:
- **删除**：删除包含缺失值的记录
- **填充**：用统计值填充（均值、中位数、众数）
- **插值**：使用插值方法填充
- **模型预测**：使用模型预测缺失值

**异常值处理**:
- **删除**：删除异常值
- **替换**：用统计值替换
- **变换**：使用变换方法处理
- **保留**：保留但标记异常值

**重复数据处理**:
- **删除**：删除重复记录
- **合并**：合并重复记录
- **标记**：标记重复记录

## 3. 数据集成

### 定义
数据集成是将来自不同数据源的数据合并为一个一致的数据集的过程。

### 集成挑战
- **模式匹配**：不同数据源的字段对应
- **数据冲突**：同一实体的不同表示
- **数据冗余**：重复或冗余的数据
- **数据不一致**：不同数据源的数据不一致

### 集成方法

**模式集成**:
- **属性匹配**：匹配不同数据源的属性
- **实体识别**：识别同一实体的不同表示
- **冲突解决**：解决数据冲突

**数据融合**:
- **数据合并**：合并不同数据源的数据
- **数据聚合**：聚合不同粒度的数据
- **数据转换**：转换数据格式和结构

## 4. 数据变换

### 定义
数据变换是将数据转换为适合机器学习算法使用的格式的过程。

### 变换类型

**数值变换**:
- **标准化**：将数据转换为均值为0，方差为1
- **归一化**：将数据缩放到[0,1]范围
- **对数变换**：使用对数函数变换
- **幂次变换**：使用幂次函数变换

**分类变换**:
- **独热编码**：将分类变量转换为二进制向量
- **标签编码**：将分类变量转换为数值
- **目标编码**：使用目标变量进行编码
- **嵌入编码**：使用嵌入向量表示

**文本变换**:
- **分词**：将文本切分为词
- **词干提取**：将词还原为词根
- **停用词去除**：去除无意义的词
- **TF-IDF**：计算词频-逆文档频率

### 变换方法

**标准化**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**归一化**:
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

**独热编码**:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
```

**标签编码**:
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

## 5. 特征工程

### 定义
特征工程是从原始数据中提取、构造和选择对机器学习模型有用的特征的过程。

### 特征类型

**原始特征**:
- 直接从数据中提取的特征
- 不需要额外处理
- 例子：年龄、收入、性别

**构造特征**:
- 通过组合原始特征构造
- 需要领域知识
- 例子：收入/年龄、BMI指数

**变换特征**:
- 通过数学变换得到
- 改善数据分布
- 例子：对数变换、平方根变换

**聚合特征**:
- 通过聚合操作得到
- 减少数据维度
- 例子：平均值、最大值、计数

### 特征构造方法

**数学运算**:
- 四则运算：加减乘除
- 幂次运算：平方、立方
- 对数运算：自然对数、常用对数
- 三角函数：正弦、余弦

**统计特征**:
- 中心趋势：均值、中位数、众数
- 离散程度：方差、标准差、极差
- 分布形状：偏度、峰度
- 分位数：四分位数、百分位数

**时间特征**:
- 时间戳：年、月、日、时
- 周期性：星期、季节
- 时间差：距离某个时间点的时间
- 滑动窗口：滑动平均值、滑动最大值

**文本特征**:
- 词频：词的出现频率
- TF-IDF：词频-逆文档频率
- 词向量：Word2Vec、GloVe
- 文本长度：字符数、词数

### 特征选择

**过滤方法**:
- 基于统计特征选择
- 计算特征与目标的相关性
- 选择相关性高的特征
- 例子：卡方检验、互信息

**包装方法**:
- 基于模型性能选择
- 使用搜索算法选择特征子集
- 计算每个子集的性能
- 例子：递归特征消除、前向选择

**嵌入方法**:
- 在模型训练过程中选择
- 使用正则化方法
- 自动选择重要特征
- 例子：L1正则化、树模型特征重要性

## 6. 数据规约

### 定义
数据规约是通过各种技术减少数据量，同时保持数据完整性的过程。

### 规约方法

**维度规约**:
- **主成分分析(PCA)**：线性降维
- **线性判别分析(LDA)**：有监督降维
- **t-SNE**：非线性降维
- **UMAP**：统一流形逼近和投影

**数值规约**:
- **离散化**：将连续值转换为离散值
- **分箱**：将连续值分组
- **聚类**：将相似值聚类
- **采样**：随机采样或分层采样

**数据压缩**:
- **无损压缩**：不丢失信息
- **有损压缩**：允许少量信息丢失
- **压缩比**：压缩后大小/原始大小

### 降维方法

**主成分分析(PCA)**:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

**线性判别分析(LDA)**:
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_reduced = lda.fit_transform(X, y)
```

**t-SNE**:
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_reduced = tsne.fit_transform(X)
```

## 7. 数据分割

### 定义
数据分割是将数据集分为训练集、验证集和测试集的过程。

### 分割策略

**简单分割**:
- 训练集：70%
- 验证集：15%
- 测试集：15%

**分层分割**:
- 保持每个集合中类别比例相同
- 适用于类别不平衡的数据
- 使用分层抽样

**时间序列分割**:
- 按时间顺序分割
- 避免数据泄露
- 适用于时间序列数据

**交叉验证**:
- K折交叉验证
- 留一交叉验证
- 分层交叉验证

### 分割方法

**随机分割**:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**分层分割**:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**K折交叉验证**:
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

## 8. 数据质量评估

### 定义
数据质量评估是评估数据质量的过程。

### 质量维度

**完整性**:
- 缺失值比例
- 数据覆盖率
- 字段完整性

**准确性**:
- 数据正确性
- 格式正确性
- 范围正确性

**一致性**:
- 数据格式一致性
- 数据值一致性
- 数据关系一致性

**及时性**:
- 数据新鲜度
- 更新频率
- 延迟时间

**有效性**:
- 数据有效性
- 业务规则符合性
- 约束条件满足性

### 评估方法

**统计方法**:
- 描述性统计
- 分布分析
- 相关性分析

**规则方法**:
- 业务规则检查
- 数据约束检查
- 格式规则检查

**机器学习方法**:
- 异常检测
- 数据漂移检测
- 质量预测

## 9. 数据预处理工具

### Python库

**pandas**:
- 数据处理和分析
- 数据清洗和变换
- 数据聚合和分组

**numpy**:
- 数值计算
- 数组操作
- 数学函数

**scikit-learn**:
- 数据预处理
- 特征工程
- 模型选择

**nltk**:
- 自然语言处理
- 文本预处理
- 语言分析

**spaCy**:
- 自然语言处理
- 文本预处理
- 语言模型

### 数据预处理流程

**1. 数据探索**:
```python
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('data.csv')

# 基本信息
print(df.info())
print(df.describe())

# 缺失值检查
print(df.isnull().sum())

# 异常值检查
print(df.boxplot())
```

**2. 数据清洗**:
```python
# 处理缺失值
df.fillna(df.mean(), inplace=True)

# 处理异常值
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 处理重复值
df.drop_duplicates(inplace=True)
```

**3. 特征工程**:
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 数值特征标准化
scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# 分类特征编码
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])
```

**4. 数据分割**:
```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## 10. 数据预处理最佳实践

### 一般原则
- **理解数据**：充分理解数据的含义和特点
- **保持一致性**：在整个流程中保持数据一致性
- **文档记录**：记录所有预处理步骤
- **版本控制**：使用版本控制管理数据

### 具体建议
- **数据探索**：先探索数据，再决定预处理策略
- **逐步处理**：逐步进行预处理，每步都验证结果
- **保留原始数据**：保留原始数据，便于回滚
- **测试验证**：使用测试集验证预处理效果

### 常见陷阱
- **数据泄露**：避免测试数据泄露到训练过程
- **过度拟合**：避免在预处理中过度拟合
- **信息丢失**：避免在预处理中丢失重要信息
- **假设错误**：避免对数据做出错误假设
