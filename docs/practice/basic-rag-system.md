# 基础RAG系统构建实战

## 项目概述

基于 [Z-RAG](https://github.com/James-Zou/Z-RAG) 的检索增强生成技术，构建一个智能问答系统，展示RAG系统的核心实现和最佳实践。

## 核心价值分析

### Z-RAG 核心实现
- **检索增强生成**: 结合信息检索和生成模型，提高回答准确性
- **多模态支持**: 支持文本、图像等多种输入类型
- **知识库管理**: 高效的文档存储和检索机制
- **生成质量控制**: 确保生成内容的准确性和相关性

### 技术架构优势
- **模块化设计**: 检索、生成、评估模块独立可扩展
- **性能优化**: 支持大规模文档检索和实时生成
- **可配置性**: 灵活的参数配置和模型选择
- **监控体系**: 完整的性能监控和质量评估

## 实战目标

构建一个企业知识库问答系统：
1. 文档检索和向量化
2. 智能问答生成
3. 答案质量评估
4. 用户交互界面

## 环境准备

### 1. 项目初始化

```bash
# 创建Python项目
mkdir rag-qa-system
cd rag-qa-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 初始化项目结构
mkdir -p src/{retrieval,generation,evaluation,api}
mkdir -p data/{documents,vectors,models}
mkdir -p tests
```

### 2. 安装依赖

```bash
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

```bash
pip install -r requirements.txt
```

## 核心实现

### 1. 文档检索模块

```python
# src/retrieval/document_processor.py
import os
import json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vector_dim = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.documents = []
        
    def load_documents(self, data_dir: str) -> List[Dict[str, Any]]:
        """加载文档数据"""
        documents = []
        data_path = Path(data_dir)
        
        for file_path in data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'id': file_path.stem,
                    'title': file_path.stem,
                    'content': content,
                    'file_path': str(file_path)
                })
        
        self.documents = documents
        return documents
    
    def create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """创建文档向量"""
        texts = [doc['content'] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """构建FAISS索引"""
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        # 归一化向量以提高余弦相似度计算
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if self.index is None:
            raise ValueError("索引未构建，请先调用build_index")
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(score),
                    'rank': len(results) + 1
                })
        
        return results
    
    def save_index(self, index_path: str):
        """保存索引"""
        if self.index is not None:
            faiss.write_index(self.index, f"{index_path}.index")
            with open(f"{index_path}_documents.json", 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def load_index(self, index_path: str):
        """加载索引"""
        self.index = faiss.read_index(f"{index_path}.index")
        with open(f"{index_path}_documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
```

### 2. 生成模块

```python
# src/generation/answer_generator.py
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AnswerGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_answer(self, 
                       question: str, 
                       context_documents: List[Dict[str, Any]], 
                       max_length: int = 512) -> str:
        """基于检索到的文档生成答案"""
        
        # 构建上下文
        context = self._build_context(context_documents)
        prompt = f"基于以下上下文回答问题：\n\n上下文：{context}\n\n问题：{question}\n\n答案："
        
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = inputs.to(self.device)
        
        # 生成答案
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("答案：")[-1].strip()
        
        return answer
    
    def _build_context(self, documents: List[Dict[str, Any]], max_context_length: int = 2000) -> str:
        """构建上下文"""
        context_parts = []
        current_length = 0
        
        for doc in documents:
            doc_text = f"文档：{doc['document']['title']}\n内容：{doc['document']['content'][:500]}..."
            if current_length + len(doc_text) > max_context_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)
```

### 3. 评估模块

```python
# src/evaluation/answer_evaluator.py
from typing import List, Dict, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AnswerEvaluator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def evaluate_answer(self, 
                       question: str, 
                       answer: str, 
                       context_documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估答案质量"""
        
        metrics = {}
        
        # 1. 相关性评分
        metrics['relevance'] = self._calculate_relevance(question, answer)
        
        # 2. 完整性评分
        metrics['completeness'] = self._calculate_completeness(question, answer)
        
        # 3. 一致性评分
        metrics['consistency'] = self._calculate_consistency(answer, context_documents)
        
        # 4. 流畅性评分
        metrics['fluency'] = self._calculate_fluency(answer)
        
        # 综合评分
        metrics['overall'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _calculate_relevance(self, question: str, answer: str) -> float:
        """计算答案与问题的相关性"""
        try:
            # 使用TF-IDF计算相似度
            texts = [question, answer]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_completeness(self, question: str, answer: str) -> float:
        """计算答案的完整性"""
        # 简单的长度和关键词检查
        min_length = 20
        if len(answer) < min_length:
            return 0.3
        
        # 检查是否包含问句中的关键词
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        
        if len(question_words) == 0:
            return 0.5
        
        overlap = len(question_words.intersection(answer_words))
        completeness = min(overlap / len(question_words), 1.0)
        
        return completeness
    
    def _calculate_consistency(self, answer: str, context_documents: List[Dict[str, Any]]) -> float:
        """计算答案与上下文的一致性"""
        if not context_documents:
            return 0.5
        
        try:
            # 提取上下文内容
            context_texts = [doc['document']['content'][:500] for doc in context_documents]
            context_texts.append(answer)
            
            # 计算TF-IDF相似度
            tfidf_matrix = self.vectorizer.fit_transform(context_texts)
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
            
            return float(np.mean(similarities))
        except:
            return 0.5
    
    def _calculate_fluency(self, answer: str) -> float:
        """计算答案的流畅性"""
        # 简单的流畅性检查
        sentences = re.split(r'[.!?]+', answer)
        if len(sentences) < 2:
            return 0.5
        
        # 检查句子长度分布
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0.5
        
        # 理想句子长度在5-25词之间
        ideal_lengths = [5 <= length <= 25 for length in sentence_lengths]
        fluency = sum(ideal_lengths) / len(ideal_lengths)
        
        return fluency
```

### 4. RAG系统集成

```python
# src/rag_system.py
from typing import List, Dict, Any, Optional
from retrieval.document_processor import DocumentProcessor
from generation.answer_generator import AnswerGenerator
from evaluation.answer_evaluator import AnswerEvaluator

class RAGSystem:
    def __init__(self, 
                 retrieval_model: str = "all-MiniLM-L6-v2",
                 generation_model: str = "microsoft/DialoGPT-medium"):
        self.retriever = DocumentProcessor(retrieval_model)
        self.generator = AnswerGenerator(generation_model)
        self.evaluator = AnswerEvaluator()
        self.is_initialized = False
    
    def initialize(self, data_dir: str, index_path: Optional[str] = None):
        """初始化RAG系统"""
        if index_path and self._load_existing_index(index_path):
            print("加载现有索引...")
        else:
            print("构建新索引...")
            documents = self.retriever.load_documents(data_dir)
            embeddings = self.retriever.create_embeddings(documents)
            self.retriever.build_index(embeddings)
            
            # 保存索引
            if index_path:
                self.retriever.save_index(index_path)
        
        self.is_initialized = True
        print("RAG系统初始化完成")
    
    def _load_existing_index(self, index_path: str) -> bool:
        """加载现有索引"""
        try:
            self.retriever.load_index(index_path)
            return True
        except:
            return False
    
    def ask_question(self, 
                    question: str, 
                    top_k: int = 5,
                    evaluate: bool = True) -> Dict[str, Any]:
        """回答问题"""
        if not self.is_initialized:
            raise ValueError("RAG系统未初始化，请先调用initialize方法")
        
        # 1. 检索相关文档
        relevant_docs = self.retriever.search(question, top_k)
        
        # 2. 生成答案
        answer = self.generator.generate_answer(question, relevant_docs)
        
        # 3. 评估答案质量
        evaluation = None
        if evaluate:
            evaluation = self.evaluator.evaluate_answer(question, answer, relevant_docs)
        
        return {
            'question': question,
            'answer': answer,
            'relevant_documents': relevant_docs,
            'evaluation': evaluation
        }
    
    def batch_questions(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量处理问题"""
        results = []
        for question in questions:
            result = self.ask_question(question, **kwargs)
            results.append(result)
        return results
```

### 5. API接口

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_system import RAGSystem

app = FastAPI(title="RAG问答系统", version="1.0.0")

# 全局RAG系统实例
rag_system = None

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    evaluate: Optional[bool] = True

class QuestionResponse(BaseModel):
    question: str
    answer: str
    relevant_documents: List[dict]
    evaluation: Optional[dict]

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = RAGSystem()
    # 初始化系统（需要提供数据目录）
    # rag_system.initialize("data/documents", "data/vectors/index")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not rag_system or not rag_system.is_initialized:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")
    
    try:
        result = rag_system.ask_question(
            question=request.question,
            top_k=request.top_k,
            evaluate=request.evaluate
        )
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "initialized": rag_system.is_initialized if rag_system else False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 测试验证

### 1. 准备测试数据

```bash
# 创建测试文档
mkdir -p data/documents
echo "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。" > data/documents/ai_basics.txt
echo "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。" > data/documents/ml_basics.txt
echo "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂模式。" > data/documents/dl_basics.txt
```

### 2. 运行系统

```python
# test_rag_system.py
from src.rag_system import RAGSystem

# 初始化系统
rag = RAGSystem()
rag.initialize("data/documents", "data/vectors/index")

# 测试问答
questions = [
    "什么是人工智能？",
    "机器学习和深度学习有什么区别？",
    "如何训练一个深度学习模型？"
]

for question in questions:
    result = rag.ask_question(question)
    print(f"问题: {result['question']}")
    print(f"答案: {result['answer']}")
    print(f"评分: {result['evaluation']['overall']:.2f}")
    print("-" * 50)
```

### 3. 启动API服务

```bash
python src/api/main.py
```

### 4. 测试API

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是人工智能？", "top_k": 3}'
```

## 核心能力总结

通过本实战项目，我们展示了以下核心能力：

1. **文档检索**: 高效的向量化检索和相似度计算
2. **答案生成**: 基于上下文的智能答案生成
3. **质量评估**: 多维度答案质量评估体系
4. **系统集成**: 模块化的RAG系统架构
5. **API服务**: 标准化的RESTful接口

## 扩展方向

1. **多模态支持**: 支持图像、音频等多种输入
2. **实时更新**: 支持知识库的实时更新
3. **个性化**: 基于用户历史的个性化推荐
4. **多语言**: 支持多语言问答
5. **可视化**: 添加检索和生成过程的可视化

## 学习要点

- 理解RAG系统的核心原理
- 掌握向量检索和相似度计算
- 学习生成模型的微调和优化
- 了解答案质量评估方法
- 实践微服务架构设计

---

**通过这个实战项目，你将掌握RAG系统开发的核心技能！**
