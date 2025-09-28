# AI安全防护实战

## 项目概述

基于 [ZSGuardian](https://github.com/James-Zou/ZSGuardian) 的安全防护能力，构建一个AI模型安全监控和防护系统，展示AI安全的核心实现和最佳实践。

## 核心价值分析

### ZSGuardian 核心实现
- **零样本安全检测**: 检测未知威胁和恶意输入
- **模型鲁棒性增强**: 提高模型对对抗性攻击的抵抗能力
- **实时监控**: 持续监控模型输入和输出
- **异常检测**: 识别异常行为和潜在威胁

### 安全防护优势
- **主动防护**: 在攻击发生前进行预防
- **自适应学习**: 根据新威胁动态调整防护策略
- **多维度检测**: 从多个角度评估输入安全性
- **可解释性**: 提供安全决策的可解释性

## 实战目标

构建一个AI模型安全防护系统：
1. 输入验证和清洗
2. 对抗性攻击检测
3. 模型输出监控
4. 安全事件响应

## 环境准备

### 1. 项目初始化

```bash
# 创建Python项目
mkdir ai-security-system
cd ai-security-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 初始化项目结构
mkdir -p src/{detection,protection,monitoring,response}
mkdir -p data/{models,logs,threats}
mkdir -p tests
```

### 2. 安装依赖

```bash
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
loguru>=0.7.0
prometheus-client>=0.17.0
```

```bash
pip install -r requirements.txt
```

## 核心实现

### 1. 输入检测模块

```python
# src/detection/input_detector.py
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class InputDetector:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        # 初始化异常检测器
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 威胁模式
        self.threat_patterns = [
            r'(?i)(hack|exploit|attack|malware|virus)',
            r'(?i)(inject|sql|script|xss)',
            r'(?i)(bypass|circumvent|evade)',
            r'(?i)(admin|root|privilege|escalation)',
            r'(?i)(backdoor|trojan|keylogger)',
            r'(?i)(ddos|dos|flood|overload)',
            r'(?i)(phishing|scam|fraud|fake)',
            r'(?i)(steal|theft|breach|leak)'
        ]
    
    def detect_malicious_input(self, text: str) -> Dict[str, Any]:
        """检测恶意输入"""
        results = {
            'is_malicious': False,
            'threat_level': 'low',
            'detected_patterns': [],
            'confidence': 0.0,
            'risk_factors': []
        }
        
        # 1. 模式匹配检测
        pattern_matches = self._detect_patterns(text)
        if pattern_matches:
            results['detected_patterns'] = pattern_matches
            results['is_malicious'] = True
            results['threat_level'] = 'high'
            results['confidence'] = 0.9
        
        # 2. 异常检测
        anomaly_score = self._detect_anomaly(text)
        if anomaly_score > 0.7:
            results['is_malicious'] = True
            results['threat_level'] = 'medium'
            results['confidence'] = max(results['confidence'], anomaly_score)
            results['risk_factors'].append('anomalous_pattern')
        
        # 3. 长度和复杂度检测
        complexity_score = self._analyze_complexity(text)
        if complexity_score > 0.8:
            results['risk_factors'].append('high_complexity')
            if not results['is_malicious']:
                results['threat_level'] = 'medium'
                results['confidence'] = complexity_score
        
        # 4. 编码检测
        encoding_issues = self._detect_encoding_issues(text)
        if encoding_issues:
            results['risk_factors'].append('suspicious_encoding')
            results['confidence'] = max(results['confidence'], 0.6)
        
        return results
    
    def _detect_patterns(self, text: str) -> List[str]:
        """检测威胁模式"""
        detected = []
        for pattern in self.threat_patterns:
            if re.search(pattern, text):
                detected.append(pattern)
        return detected
    
    def _detect_anomaly(self, text: str) -> float:
        """异常检测"""
        if not self.is_fitted:
            return 0.0
        
        try:
            # 提取特征
            features = self._extract_features(text)
            features = self.scaler.transform([features])
            
            # 异常检测
            anomaly_score = self.anomaly_detector.decision_function(features)[0]
            # 转换为0-1范围
            normalized_score = max(0, min(1, (anomaly_score + 0.5) / 1.0))
            return normalized_score
        except:
            return 0.0
    
    def _analyze_complexity(self, text: str) -> float:
        """分析文本复杂度"""
        # 长度因子
        length_factor = min(len(text) / 1000, 1.0)
        
        # 特殊字符比例
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        special_ratio = special_chars / max(len(text), 1)
        
        # 重复字符
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        max_repetition = max(char_counts.values()) / len(text) if char_counts else 0
        
        # 综合复杂度
        complexity = (length_factor * 0.3 + special_ratio * 0.4 + max_repetition * 0.3)
        return min(complexity, 1.0)
    
    def _detect_encoding_issues(self, text: str) -> bool:
        """检测编码问题"""
        try:
            # 检查是否包含可疑的编码模式
            suspicious_patterns = [
                r'%[0-9A-Fa-f]{2}',  # URL编码
                r'\\x[0-9A-Fa-f]{2}',  # 十六进制编码
                r'\\u[0-9A-Fa-f]{4}',  # Unicode编码
                r'&#[0-9]+;',  # HTML实体编码
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, text):
                    return True
            return False
        except:
            return False
    
    def _extract_features(self, text: str) -> List[float]:
        """提取文本特征"""
        features = []
        
        # 基础特征
        features.append(len(text))  # 长度
        features.append(len(text.split()))  # 词数
        features.append(len(set(text.split())))  # 唯一词数
        
        # 字符特征
        features.append(len(re.findall(r'[a-zA-Z]', text)))  # 字母数
        features.append(len(re.findall(r'[0-9]', text)))  # 数字数
        features.append(len(re.findall(r'[^a-zA-Z0-9\s]', text)))  # 特殊字符数
        
        # 统计特征
        features.append(np.mean([len(word) for word in text.split()]))  # 平均词长
        features.append(len(re.findall(r'[A-Z]', text)) / max(len(text), 1))  # 大写比例
        
        return features
    
    def fit_anomaly_detector(self, training_texts: List[str]):
        """训练异常检测器"""
        # 提取特征
        features = [self._extract_features(text) for text in training_texts]
        features = np.array(features)
        
        # 标准化特征
        features = self.scaler.fit_transform(features)
        
        # 训练异常检测器
        self.anomaly_detector.fit(features)
        self.is_fitted = True
```

### 2. 对抗性攻击检测

```python
# src/detection/adversarial_detector.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class AdversarialDetector:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 对抗性模式
        self.adversarial_patterns = [
            r'[^\x00-\x7F]',  # 非ASCII字符
            r'[^\w\s]',  # 特殊字符
            r'(.)\1{3,}',  # 重复字符
            r'\s{3,}',  # 多个空格
        ]
    
    def detect_adversarial_attack(self, 
                                original_text: str, 
                                modified_text: str) -> Dict[str, Any]:
        """检测对抗性攻击"""
        results = {
            'is_adversarial': False,
            'attack_type': 'unknown',
            'confidence': 0.0,
            'similarity': 0.0,
            'detected_changes': []
        }
        
        # 1. 计算语义相似度
        similarity = self._calculate_semantic_similarity(original_text, modified_text)
        results['similarity'] = similarity
        
        # 2. 检测字符级变化
        char_changes = self._detect_character_changes(original_text, modified_text)
        if char_changes['change_ratio'] > 0.1:
            results['is_adversarial'] = True
            results['attack_type'] = 'character_substitution'
            results['confidence'] = char_changes['change_ratio']
            results['detected_changes'].append('character_changes')
        
        # 3. 检测模式变化
        pattern_changes = self._detect_pattern_changes(original_text, modified_text)
        if pattern_changes['pattern_score'] > 0.5:
            results['is_adversarial'] = True
            results['attack_type'] = 'pattern_manipulation'
            results['confidence'] = max(results['confidence'], pattern_changes['pattern_score'])
            results['detected_changes'].append('pattern_changes')
        
        # 4. 检测语义变化
        if similarity < 0.7:
            results['is_adversarial'] = True
            results['attack_type'] = 'semantic_manipulation'
            results['confidence'] = max(results['confidence'], 1 - similarity)
            results['detected_changes'].append('semantic_changes')
        
        return results
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        try:
            # 编码文本
            inputs1 = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
            inputs2 = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
            
            inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
            
            # 获取嵌入
            with torch.no_grad():
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
                
                # 使用[CLS]标记的嵌入
                embedding1 = outputs1.last_hidden_state[:, 0, :].cpu().numpy()
                embedding2 = outputs2.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 计算余弦相似度
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _detect_character_changes(self, text1: str, text2: str) -> Dict[str, Any]:
        """检测字符级变化"""
        if len(text1) != len(text2):
            return {'change_ratio': 1.0, 'changes': []}
        
        changes = []
        for i, (c1, c2) in enumerate(zip(text1, text2)):
            if c1 != c2:
                changes.append({'position': i, 'original': c1, 'modified': c2})
        
        change_ratio = len(changes) / len(text1) if text1 else 0
        return {'change_ratio': change_ratio, 'changes': changes}
    
    def _detect_pattern_changes(self, text1: str, text2: str) -> Dict[str, Any]:
        """检测模式变化"""
        pattern_scores = []
        
        for pattern in self.adversarial_patterns:
            matches1 = len(re.findall(pattern, text1))
            matches2 = len(re.findall(pattern, text2))
            
            if matches1 > 0 or matches2 > 0:
                score = abs(matches2 - matches1) / max(matches1 + matches2, 1)
                pattern_scores.append(score)
        
        pattern_score = np.mean(pattern_scores) if pattern_scores else 0
        return {'pattern_score': pattern_score, 'scores': pattern_scores}
```

### 3. 模型输出监控

```python
# src/monitoring/output_monitor.py
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta

class OutputMonitor:
    def __init__(self, window_size: int = 1000, alert_threshold: float = 0.8):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # 监控数据
        self.output_history = deque(maxlen=window_size)
        self.anomaly_scores = deque(maxlen=window_size)
        self.alert_history = []
        
        # 统计信息
        self.stats = {
            'total_outputs': 0,
            'anomaly_count': 0,
            'alert_count': 0,
            'avg_confidence': 0.0,
            'avg_response_time': 0.0
        }
        
        # 基线数据
        self.baseline_confidence = 0.5
        self.baseline_response_time = 1.0
    
    def monitor_output(self, 
                      output: str, 
                      confidence: float, 
                      response_time: float,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """监控模型输出"""
        timestamp = datetime.now()
        
        # 记录输出
        output_record = {
            'timestamp': timestamp,
            'output': output,
            'confidence': confidence,
            'response_time': response_time,
            'metadata': metadata or {}
        }
        
        self.output_history.append(output_record)
        
        # 计算异常分数
        anomaly_score = self._calculate_anomaly_score(output_record)
        self.anomaly_scores.append(anomaly_score)
        
        # 更新统计信息
        self._update_stats(output_record, anomaly_score)
        
        # 检查是否需要告警
        alert = self._check_alert(output_record, anomaly_score)
        
        return {
            'output_record': output_record,
            'anomaly_score': anomaly_score,
            'alert': alert,
            'stats': self.get_current_stats()
        }
    
    def _calculate_anomaly_score(self, output_record: Dict[str, Any]) -> float:
        """计算异常分数"""
        scores = []
        
        # 1. 置信度异常
        conf_score = abs(output_record['confidence'] - self.baseline_confidence)
        scores.append(conf_score)
        
        # 2. 响应时间异常
        time_score = abs(output_record['response_time'] - self.baseline_response_time) / self.baseline_response_time
        scores.append(time_score)
        
        # 3. 输出长度异常
        output_length = len(output_record['output'])
        if len(self.output_history) > 1:
            avg_length = np.mean([len(record['output']) for record in list(self.output_history)[-10:]])
            length_score = abs(output_length - avg_length) / max(avg_length, 1)
        else:
            length_score = 0
        scores.append(length_score)
        
        # 4. 输出内容异常
        content_score = self._analyze_content_anomaly(output_record['output'])
        scores.append(content_score)
        
        # 综合异常分数
        anomaly_score = np.mean(scores)
        return min(anomaly_score, 1.0)
    
    def _analyze_content_anomaly(self, output: str) -> float:
        """分析内容异常"""
        # 检查重复内容
        words = output.split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_score = 1 - (unique_words / len(words))
        else:
            repetition_score = 0
        
        # 检查异常字符
        special_chars = len([c for c in output if not c.isalnum() and not c.isspace()])
        special_ratio = special_chars / max(len(output), 1)
        
        # 检查长度异常
        length_score = 0
        if len(output) > 1000 or len(output) < 10:
            length_score = 0.5
        
        return np.mean([repetition_score, special_ratio, length_score])
    
    def _check_alert(self, output_record: Dict[str, Any], anomaly_score: float) -> Optional[Dict[str, Any]]:
        """检查是否需要告警"""
        if anomaly_score > self.alert_threshold:
            alert = {
                'timestamp': output_record['timestamp'],
                'type': 'anomaly_detected',
                'severity': 'high' if anomaly_score > 0.9 else 'medium',
                'anomaly_score': anomaly_score,
                'output_preview': output_record['output'][:100],
                'confidence': output_record['confidence'],
                'response_time': output_record['response_time']
            }
            
            self.alert_history.append(alert)
            self.stats['alert_count'] += 1
            
            return alert
        
        return None
    
    def _update_stats(self, output_record: Dict[str, Any], anomaly_score: float):
        """更新统计信息"""
        self.stats['total_outputs'] += 1
        
        if anomaly_score > 0.5:
            self.stats['anomaly_count'] += 1
        
        # 更新平均置信度
        total_conf = sum(record['confidence'] for record in self.output_history)
        self.stats['avg_confidence'] = total_conf / len(self.output_history)
        
        # 更新平均响应时间
        total_time = sum(record['response_time'] for record in self.output_history)
        self.stats['avg_response_time'] = total_time / len(self.output_history)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        return {
            **self.stats,
            'anomaly_rate': self.stats['anomaly_count'] / max(self.stats['total_outputs'], 1),
            'alert_rate': self.stats['alert_count'] / max(self.stats['total_outputs'], 1),
            'recent_alerts': len([a for a in self.alert_history 
                                if a['timestamp'] > datetime.now() - timedelta(hours=1)])
        }
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert['timestamp'] > cutoff_time]
```

### 4. 安全响应系统

```python
# src/response/security_responder.py
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from datetime import datetime

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityResponder:
    def __init__(self):
        self.response_rules = {
            ThreatLevel.LOW: self._handle_low_threat,
            ThreatLevel.MEDIUM: self._handle_medium_threat,
            ThreatLevel.HIGH: self._handle_high_threat,
            ThreatLevel.CRITICAL: self._handle_critical_threat
        }
        
        self.blocked_ips = set()
        self.blocked_patterns = set()
        self.rate_limits = {}
    
    def respond_to_threat(self, 
                         threat_info: Dict[str, Any], 
                         client_ip: Optional[str] = None) -> Dict[str, Any]:
        """响应安全威胁"""
        threat_level = self._assess_threat_level(threat_info)
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'threat_level': threat_level.value,
            'actions_taken': [],
            'blocked': False,
            'message': ''
        }
        
        # 执行响应规则
        if threat_level in self.response_rules:
            rule_response = self.response_rules[threat_level](threat_info, client_ip)
            response.update(rule_response)
        
        # 记录响应
        self._log_response(response, threat_info)
        
        return response
    
    def _assess_threat_level(self, threat_info: Dict[str, Any]) -> ThreatLevel:
        """评估威胁等级"""
        confidence = threat_info.get('confidence', 0.0)
        threat_type = threat_info.get('threat_type', 'unknown')
        
        if confidence >= 0.9 or threat_type in ['critical', 'malware', 'exploit']:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.7 or threat_type in ['high', 'adversarial']:
            return ThreatLevel.HIGH
        elif confidence >= 0.5 or threat_type in ['medium', 'suspicious']:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _handle_low_threat(self, threat_info: Dict[str, Any], client_ip: Optional[str]) -> Dict[str, Any]:
        """处理低威胁"""
        return {
            'actions_taken': ['log_threat'],
            'blocked': False,
            'message': '威胁已记录，继续处理请求'
        }
    
    def _handle_medium_threat(self, threat_info: Dict[str, Any], client_ip: Optional[str]) -> Dict[str, Any]:
        """处理中等威胁"""
        actions = ['log_threat', 'rate_limit']
        
        # 检查是否需要限制频率
        if client_ip:
            self._apply_rate_limit(client_ip, 60)  # 1分钟限制
        
        return {
            'actions_taken': actions,
            'blocked': False,
            'message': '检测到可疑活动，已应用频率限制'
        }
    
    def _handle_high_threat(self, threat_info: Dict[str, Any], client_ip: Optional[str]) -> Dict[str, Any]:
        """处理高威胁"""
        actions = ['log_threat', 'block_pattern']
        
        # 阻止恶意模式
        if 'detected_patterns' in threat_info:
            for pattern in threat_info['detected_patterns']:
                self.blocked_patterns.add(pattern)
        
        # 临时阻止IP
        if client_ip:
            self.blocked_ips.add(client_ip)
            actions.append('block_ip')
        
        return {
            'actions_taken': actions,
            'blocked': True,
            'message': '检测到高风险威胁，请求已被阻止'
        }
    
    def _handle_critical_threat(self, threat_info: Dict[str, Any], client_ip: Optional[str]) -> Dict[str, Any]:
        """处理严重威胁"""
        actions = ['log_threat', 'block_pattern', 'block_ip', 'alert_admin']
        
        # 阻止所有相关模式
        if 'detected_patterns' in threat_info:
            for pattern in threat_info['detected_patterns']:
                self.blocked_patterns.add(pattern)
        
        # 永久阻止IP
        if client_ip:
            self.blocked_ips.add(client_ip)
        
        return {
            'actions_taken': actions,
            'blocked': True,
            'message': '检测到严重威胁，已采取紧急防护措施'
        }
    
    def _apply_rate_limit(self, client_ip: str, duration_seconds: int):
        """应用频率限制"""
        self.rate_limits[client_ip] = {
            'start_time': datetime.now(),
            'duration': duration_seconds,
            'request_count': 0
        }
    
    def is_blocked(self, client_ip: Optional[str], input_text: str) -> bool:
        """检查是否被阻止"""
        # 检查IP是否被阻止
        if client_ip and client_ip in self.blocked_ips:
            return True
        
        # 检查是否在频率限制期内
        if client_ip and client_ip in self.rate_limits:
            rate_limit = self.rate_limits[client_ip]
            if (datetime.now() - rate_limit['start_time']).seconds < rate_limit['duration']:
                return True
        
        # 检查输入是否包含被阻止的模式
        for pattern in self.blocked_patterns:
            if pattern in input_text:
                return True
        
        return False
    
    def _log_response(self, response: Dict[str, Any], threat_info: Dict[str, Any]):
        """记录响应日志"""
        log_entry = {
            'timestamp': response['timestamp'],
            'threat_level': response['threat_level'],
            'threat_info': threat_info,
            'response': response
        }
        
        # 这里可以集成到实际的日志系统
        print(f"Security Response: {json.dumps(log_entry, indent=2)}")
```

### 5. 安全系统集成

```python
# src/ai_security_system.py
from typing import Dict, Any, Optional
from detection.input_detector import InputDetector
from detection.adversarial_detector import AdversarialDetector
from monitoring.output_monitor import OutputMonitor
from response.security_responder import SecurityResponder
import time

class AISecuritySystem:
    def __init__(self):
        self.input_detector = InputDetector()
        self.adversarial_detector = AdversarialDetector()
        self.output_monitor = OutputMonitor()
        self.security_responder = SecurityResponder()
        
        # 训练异常检测器（使用示例数据）
        self._train_anomaly_detector()
    
    def _train_anomaly_detector(self):
        """训练异常检测器"""
        # 使用一些示例数据训练
        training_texts = [
            "Hello, how are you?",
            "What is the weather like today?",
            "Can you help me with this problem?",
            "I need assistance with my account.",
            "Thank you for your help."
        ]
        self.input_detector.fit_anomaly_detector(training_texts)
    
    def process_request(self, 
                       input_text: str, 
                       client_ip: Optional[str] = None,
                       original_text: Optional[str] = None) -> Dict[str, Any]:
        """处理请求的安全检查"""
        start_time = time.time()
        
        # 1. 检查是否被阻止
        if self.security_responder.is_blocked(client_ip, input_text):
            return {
                'blocked': True,
                'reason': 'Request blocked by security system',
                'message': 'Your request has been blocked due to security concerns.'
            }
        
        # 2. 输入检测
        input_detection = self.input_detector.detect_malicious_input(input_text)
        
        # 3. 对抗性攻击检测
        adversarial_detection = None
        if original_text and original_text != input_text:
            adversarial_detection = self.adversarial_detector.detect_adversarial_attack(
                original_text, input_text
            )
        
        # 4. 综合威胁评估
        threat_info = self._assess_overall_threat(input_detection, adversarial_detection)
        
        # 5. 安全响应
        security_response = self.security_responder.respond_to_threat(threat_info, client_ip)
        
        # 6. 如果被阻止，直接返回
        if security_response.get('blocked', False):
            return {
                'blocked': True,
                'reason': security_response['message'],
                'threat_level': threat_info.get('threat_level', 'unknown'),
                'actions_taken': security_response.get('actions_taken', [])
            }
        
        # 7. 处理请求（这里可以调用实际的AI模型）
        response_time = time.time() - start_time
        ai_response = self._generate_ai_response(input_text)
        
        # 8. 输出监控
        output_monitoring = self.output_monitor.monitor_output(
            output=ai_response,
            confidence=0.8,  # 这里应该是实际的置信度
            response_time=response_time,
            metadata={'input_detection': input_detection}
        )
        
        return {
            'blocked': False,
            'response': ai_response,
            'security_info': {
                'input_detection': input_detection,
                'adversarial_detection': adversarial_detection,
                'threat_assessment': threat_info,
                'security_response': security_response,
                'output_monitoring': output_monitoring
            }
        }
    
    def _assess_overall_threat(self, 
                              input_detection: Dict[str, Any], 
                              adversarial_detection: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """综合评估威胁"""
        threat_level = 'low'
        confidence = 0.0
        
        # 基于输入检测
        if input_detection.get('is_malicious', False):
            threat_level = input_detection.get('threat_level', 'medium')
            confidence = max(confidence, input_detection.get('confidence', 0.0))
        
        # 基于对抗性检测
        if adversarial_detection and adversarial_detection.get('is_adversarial', False):
            if threat_level == 'low':
                threat_level = 'medium'
            confidence = max(confidence, adversarial_detection.get('confidence', 0.0))
        
        return {
            'threat_level': threat_level,
            'confidence': confidence,
            'input_detection': input_detection,
            'adversarial_detection': adversarial_detection
        }
    
    def _generate_ai_response(self, input_text: str) -> str:
        """生成AI响应（模拟）"""
        # 这里应该调用实际的AI模型
        return f"AI Response to: {input_text[:50]}..."
    
    def get_security_stats(self) -> Dict[str, Any]:
        """获取安全统计信息"""
        return {
            'output_monitor_stats': self.output_monitor.get_current_stats(),
            'blocked_ips_count': len(self.security_responder.blocked_ips),
            'blocked_patterns_count': len(self.security_responder.blocked_patterns),
            'rate_limited_ips': len(self.security_responder.rate_limits)
        }
```

## 测试验证

### 1. 基本功能测试

```python
# test_security_system.py
from src.ai_security_system import AISecuritySystem

# 初始化安全系统
security_system = AISecuritySystem()

# 测试正常请求
normal_request = "Hello, how are you today?"
result = security_system.process_request(normal_request, "192.168.1.100")
print("正常请求结果:", result['blocked'])

# 测试恶意请求
malicious_request = "I want to hack your system and steal data"
result = security_system.process_request(malicious_request, "192.168.1.101")
print("恶意请求结果:", result['blocked'])

# 测试对抗性攻击
original = "What is the weather like?"
adversarial = "What is the w3ather l1ke?"  # 字符替换
result = security_system.process_request(adversarial, "192.168.1.102", original)
print("对抗性攻击结果:", result['blocked'])
```

### 2. 性能测试

```python
# performance_test.py
import time
import random
from src.ai_security_system import AISecuritySystem

def performance_test():
    security_system = AISecuritySystem()
    
    # 生成测试数据
    test_cases = [
        "Hello, how are you?",
        "What is the weather like?",
        "I need help with my account",
        "This is a normal question",
        "Can you assist me please?"
    ]
    
    # 性能测试
    start_time = time.time()
    for i in range(1000):
        text = random.choice(test_cases)
        result = security_system.process_request(text, f"192.168.1.{i % 255}")
    
    end_time = time.time()
    print(f"处理1000个请求耗时: {end_time - start_time:.2f}秒")
    print(f"平均每个请求: {(end_time - start_time) / 1000 * 1000:.2f}毫秒")

if __name__ == "__main__":
    performance_test()
```

## 核心能力总结

通过本实战项目，我们展示了以下核心能力：

1. **输入安全检测**: 多维度检测恶意输入和异常模式
2. **对抗性攻击防护**: 检测和防护各种对抗性攻击
3. **输出监控**: 实时监控模型输出质量和异常
4. **智能响应**: 基于威胁等级的自适应响应机制
5. **系统集成**: 完整的安全防护体系

## 扩展方向

1. **机器学习增强**: 使用ML模型提高检测准确性
2. **实时学习**: 根据新威胁动态更新防护策略
3. **可视化监控**: 添加安全监控的可视化界面
4. **分布式部署**: 支持大规模分布式安全防护
5. **合规性**: 添加安全合规性检查和报告

## 学习要点

- 理解AI安全的核心威胁和防护策略
- 掌握输入验证和异常检测技术
- 学习对抗性攻击的检测和防护
- 了解安全监控和响应机制
- 实践安全系统的设计和实现

---

**通过这个实战项目，你将掌握AI安全防护的核心技能！**
