# 特征工程

## 1. 特征工程概述

### 定义
特征工程是从原始数据中提取、构造和选择对机器学习模型有用的特征的过程。

### 重要性
- **数据质量决定模型上限**：好的特征比复杂的算法更重要
- **提高模型性能**：合适的特征能显著提升模型效果
- **降低计算复杂度**：减少特征维度，提高效率
- **增强可解释性**：有意义的特征便于理解模型

### 特征工程流程
1. 特征理解
2. 特征提取
3. 特征构造
4. 特征选择
5. 特征变换
6. 特征验证

## 2. 特征类型

### 按数据类型分类

**数值特征**:
- 连续数值：年龄、收入、温度
- 离散数值：计数、排名、等级
- 特点：可以进行数学运算

**分类特征**:
- 标称特征：颜色、性别、品牌
- 序数特征：等级、评分、优先级
- 特点：有限个离散值

**文本特征**:
- 短文本：标题、标签
- 长文本：文章、评论
- 特点：需要特殊处理

**时间特征**:
- 时间戳：日期、时间
- 时间序列：股票价格、传感器数据
- 特点：具有时间顺序性

**图像特征**:
- 像素值：原始像素
- 统计特征：均值、方差
- 深度学习特征：CNN特征
- 特点：高维、空间相关

### 按特征来源分类

**原始特征**:
- 直接从数据中提取
- 不需要额外处理
- 例子：年龄、性别、收入

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

## 3. 特征提取

### 定义
特征提取是从原始数据中提取有用信息的过程。

### 数值特征提取

**统计特征**:
- 中心趋势：均值、中位数、众数
- 离散程度：方差、标准差、极差
- 分布形状：偏度、峰度
- 分位数：四分位数、百分位数

**时间特征**:
- 时间戳：年、月、日、时、分、秒
- 周期性：星期、季节、节假日
- 时间差：距离某个时间点的时间
- 滑动窗口：滑动平均值、滑动最大值

**序列特征**:
- 趋势：上升、下降、平稳
- 周期性：周期性模式
- 异常：异常值、突变点
- 自相关：自相关系数

### 文本特征提取

**词频特征**:
- 词频：词的出现频率
- TF-IDF：词频-逆文档频率
- 词数：总词数、唯一词数
- 句子数：总句子数、平均句子长度

**词向量特征**:
- Word2Vec：词向量表示
- GloVe：全局词向量
- FastText：子词词向量
- 预训练词向量：BERT、GPT

**语义特征**:
- 情感分析：情感极性、情感强度
- 主题建模：LDA、LSA
- 命名实体：人名、地名、机构名
- 关键词：关键词提取

### 图像特征提取

**传统特征**:
- 颜色特征：颜色直方图、颜色矩
- 纹理特征：LBP、Gabor滤波器
- 形状特征：轮廓、面积、周长
- 边缘特征：边缘方向、边缘强度

**深度学习特征**:
- CNN特征：卷积层输出
- 预训练特征：VGG、ResNet特征
- 注意力特征：注意力权重
- 多尺度特征：不同尺度的特征

## 4. 特征构造

### 定义
特征构造是通过组合、变换现有特征来创建新特征的过程。

### 数学运算

**四则运算**:
- 加法：特征1 + 特征2
- 减法：特征1 - 特征2
- 乘法：特征1 × 特征2
- 除法：特征1 / 特征2

**幂次运算**:
- 平方：特征²
- 立方：特征³
- 平方根：√特征
- 立方根：∛特征

**对数运算**:
- 自然对数：ln(特征)
- 常用对数：log10(特征)
- 对数变换：log(特征 + 1)

**三角函数**:
- 正弦：sin(特征)
- 余弦：cos(特征)
- 正切：tan(特征)

### 统计特征构造

**滑动窗口特征**:
- 滑动平均值：rolling_mean
- 滑动最大值：rolling_max
- 滑动最小值：rolling_min
- 滑动标准差：rolling_std

**分组聚合特征**:
- 分组平均值：groupby.mean()
- 分组最大值：groupby.max()
- 分组计数：groupby.count()
- 分组求和：groupby.sum()

**时间特征构造**:
- 时间差：当前时间 - 历史时间
- 周期性特征：sin(2π * 时间 / 周期)
- 趋势特征：线性回归斜率
- 季节性特征：季节性分解

### 交互特征构造

**特征组合**:
- 特征拼接：concat(feature1, feature2)
- 特征交叉：feature1 × feature2
- 特征比值：feature1 / feature2
- 特征差值：feature1 - feature2

**多项式特征**:
- 二次项：feature²
- 交叉项：feature1 × feature2
- 高次项：feature³
- 多项式组合：feature1² + feature2²

**分箱特征**:
- 等宽分箱：等宽度区间
- 等频分箱：等频率区间
- 自定义分箱：业务规则分箱
- 分箱编码：独热编码

## 5. 特征选择

### 定义
特征选择是从所有特征中选择最重要的特征子集的过程。

### 选择目标
- 提高模型性能
- 减少过拟合
- 降低计算复杂度
- 提高可解释性

### 选择方法

**过滤方法**:
- 基于统计特征选择
- 计算特征与目标的相关性
- 选择相关性高的特征
- 优点：计算简单，速度快
- 缺点：忽略特征间交互

**包装方法**:
- 基于模型性能选择
- 使用搜索算法选择特征子集
- 计算每个子集的性能
- 优点：考虑特征间交互
- 缺点：计算复杂度高

**嵌入方法**:
- 在模型训练过程中选择
- 使用正则化方法
- 自动选择重要特征
- 优点：结合训练过程
- 缺点：依赖特定模型

### 过滤方法

**相关性分析**:
```python
import pandas as pd
import numpy as np

# 计算相关系数
correlation = df.corr()['target'].abs().sort_values(ascending=False)

# 选择相关性高的特征
selected_features = correlation[correlation > 0.1].index.tolist()
```

**卡方检验**:
```python
from sklearn.feature_selection import chi2, SelectKBest

# 卡方检验
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)
```

**互信息**:
```python
from sklearn.feature_selection import mutual_info_classif

# 计算互信息
mi_scores = mutual_info_classif(X, y)

# 选择互信息高的特征
selected_features = X.columns[mi_scores > 0.1]
```

### 包装方法

**递归特征消除**:
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 递归特征消除
selector = RFE(LogisticRegression(), n_features_to_select=10)
X_selected = selector.fit_transform(X, y)
```

**前向选择**:
```python
from sklearn.feature_selection import SequentialFeatureSelector

# 前向选择
selector = SequentialFeatureSelector(
    LogisticRegression(), 
    n_features_to_select=10,
    direction='forward'
)
X_selected = selector.fit_transform(X, y)
```

**后向消除**:
```python
from sklearn.feature_selection import SequentialFeatureSelector

# 后向消除
selector = SequentialFeatureSelector(
    LogisticRegression(), 
    n_features_to_select=10,
    direction='backward'
)
X_selected = selector.fit_transform(X, y)
```

### 嵌入方法

**L1正则化**:
```python
from sklearn.linear_model import Lasso

# L1正则化
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# 选择非零系数的特征
selected_features = X.columns[lasso.coef_ != 0]
```

**树模型特征重要性**:
```python
from sklearn.ensemble import RandomForestClassifier

# 随机森林
rf = RandomForestClassifier()
rf.fit(X, y)

# 选择重要性高的特征
feature_importance = rf.feature_importances_
selected_features = X.columns[feature_importance > 0.01]
```

**梯度提升特征重要性**:
```python
from sklearn.ensemble import GradientBoostingClassifier

# 梯度提升
gb = GradientBoostingClassifier()
gb.fit(X, y)

# 选择重要性高的特征
feature_importance = gb.feature_importances_
selected_features = X.columns[feature_importance > 0.01]
```

## 6. 特征变换

### 定义
特征变换是将特征转换为更适合机器学习算法的格式的过程。

### 数值特征变换

**标准化**:
```python
from sklearn.preprocessing import StandardScaler

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**归一化**:
```python
from sklearn.preprocessing import MinMaxScaler

# 归一化
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

**鲁棒缩放**:
```python
from sklearn.preprocessing import RobustScaler

# 鲁棒缩放
scaler = RobustScaler()
X_robust = scaler.fit_transform(X)
```

**对数变换**:
```python
import numpy as np

# 对数变换
X_log = np.log1p(X)  # log(1 + x)
```

**Box-Cox变换**:
```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox变换
transformer = PowerTransformer(method='box-cox')
X_transformed = transformer.fit_transform(X)
```

### 分类特征变换

**独热编码**:
```python
from sklearn.preprocessing import OneHotEncoder

# 独热编码
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
```

**标签编码**:
```python
from sklearn.preprocessing import LabelEncoder

# 标签编码
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

**目标编码**:
```python
import pandas as pd

# 目标编码
target_mean = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_mean)
```

**频率编码**:
```python
# 频率编码
category_counts = df['category'].value_counts()
df['category_freq'] = df['category'].map(category_counts)
```

### 文本特征变换

**TF-IDF**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(text_data)
```

**词向量**:
```python
from gensim.models import Word2Vec

# Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
word_vectors = model.wv
```

**预训练词向量**:
```python
import gensim.downloader as api

# 加载预训练词向量
model = api.load('glove-wiki-gigaword-100')
word_vectors = model
```

## 7. 特征工程工具

### Python库

**pandas**:
- 数据处理和分析
- 特征构造和变换
- 数据聚合和分组

**numpy**:
- 数值计算
- 数组操作
- 数学函数

**scikit-learn**:
- 特征选择
- 特征变换
- 预处理工具

**nltk**:
- 自然语言处理
- 文本特征提取
- 语言分析

**gensim**:
- 主题建模
- 词向量
- 文档相似度

### 特征工程流程

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

# 数据类型检查
print(df.dtypes)
```

**2. 特征提取**:
```python
# 数值特征
df['age_squared'] = df['age'] ** 2
df['income_log'] = np.log1p(df['income'])

# 分类特征
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'old'])

# 时间特征
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
```

**3. 特征选择**:
```python
from sklearn.feature_selection import SelectKBest, f_classif

# 选择K个最佳特征
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

**4. 特征变换**:
```python
from sklearn.preprocessing import StandardScaler

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 8. 特征工程最佳实践

### 一般原则
- **理解业务**：充分理解业务背景和需求
- **数据质量**：确保数据质量和完整性
- **特征可解释性**：选择可解释的特征
- **避免过拟合**：避免在特征工程中过拟合

### 具体建议
- **特征探索**：先探索特征，再决定工程策略
- **逐步构造**：逐步构造特征，每步都验证效果
- **特征验证**：使用交叉验证验证特征效果
- **特征文档**：记录特征的含义和构造方法

### 常见陷阱
- **数据泄露**：避免未来信息泄露到历史数据
- **过拟合**：避免在特征工程中过拟合
- **特征冗余**：避免构造冗余特征
- **假设错误**：避免对数据做出错误假设

### 特征工程检查清单
- [ ] 数据质量检查
- [ ] 特征类型识别
- [ ] 缺失值处理
- [ ] 异常值处理
- [ ] 特征提取
- [ ] 特征构造
- [ ] 特征选择
- [ ] 特征变换
- [ ] 特征验证
- [ ] 特征文档
