# 综合AI系统实战

## 项目概述

基于多个优秀开源项目的核心能力，构建一个完整的AI应用系统，展示如何在实际项目中整合MCP协议、RAG技术、安全防护等核心能力。

## 开源项目核心能力整合

### 1. mcp-springboot-server + mcp-facade-generator
- **核心能力**: 快速构建标准化AI服务接口
- **应用场景**: 服务端API开发、工具自动注册、协议标准化

### 2. Z-RAG
- **核心能力**: 检索增强生成，提高回答准确性
- **应用场景**: 智能问答、知识库查询、内容生成

### 3. ZSGuardian
- **核心能力**: AI模型安全防护和威胁检测
- **应用场景**: 输入验证、输出监控、安全响应

## 综合系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端用户界面                              │
├─────────────────────────────────────────────────────────────┤
│                    API网关层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  认证授权   │  │  限流熔断   │  │  监控日志   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    MCP服务层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  AI问答服务 │  │  文档检索   │  │  内容生成   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    RAG引擎层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  文档处理   │  │  向量检索   │  │  答案生成   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                   安全防护层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  输入检测   │  │  威胁识别   │  │  响应处理   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                   数据存储层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  知识库     │  │  向量库     │  │  日志存储   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 实战目标

构建一个企业级智能知识问答系统：
1. 支持多种文档格式的知识库管理
2. 提供智能问答和内容生成服务
3. 集成完整的安全防护体系
4. 支持MCP协议标准化接口
5. 提供实时监控和运维管理

## 环境准备

### 1. 项目初始化

```bash
# 创建综合项目
mkdir comprehensive-ai-system
cd comprehensive-ai-system

# 创建项目结构
mkdir -p {backend,frontend,data,deploy,docs}
mkdir -p backend/{src,test,config}
mkdir -p frontend/{src,public,components}
mkdir -p data/{documents,vectors,logs}
mkdir -p deploy/{docker,kubernetes,monitoring}
```

### 2. 后端技术栈

```xml
<!-- backend/pom.xml -->
<dependencies>
    <!-- Spring Boot -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-security</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    
    <!-- Spring AI MCP -->
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-mcp-spring-boot-starter</artifactId>
        <version>1.0.0-M4</version>
    </dependency>
    
    <!-- MCP Facade Generator -->
    <dependency>
        <groupId>com.unionhole</groupId>
        <artifactId>mcp-facade-generator</artifactId>
        <version>1.0.1</version>
    </dependency>
    
    <!-- 数据库 -->
    <dependency>
        <groupId>org.postgresql</groupId>
        <artifactId>postgresql</artifactId>
    </dependency>
    <dependency>
        <groupId>org.elasticsearch.client</groupId>
        <artifactId>elasticsearch-rest-high-level-client</artifactId>
    </dependency>
    
    <!-- 缓存 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-redis</artifactId>
    </dependency>
    
    <!-- 监控 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-actuator</artifactId>
    </dependency>
    <dependency>
        <groupId>io.micrometer</groupId>
        <artifactId>micrometer-registry-prometheus</artifactId>
    </dependency>
</dependencies>
```

### 3. 前端技术栈

```json
// frontend/package.json
{
  "name": "ai-system-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "axios": "^1.3.0",
    "antd": "^5.2.0",
    "echarts": "^5.4.0",
    "monaco-editor": "^0.36.0"
  }
}
```

## 核心实现

### 1. 知识库管理服务

```java
// backend/src/main/java/com/example/ai/service/KnowledgeBaseService.java
package com.example.ai.service;

import com.unionhole.mcp.annotation.MCPService;
import com.unionhole.mcp.annotation.MCPMethod;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

@Service
@MCPService(packageName = "com.example.ai.mcp")
public class KnowledgeBaseService {
    
    @Autowired
    private DocumentProcessor documentProcessor;
    
    @Autowired
    private VectorStore vectorStore;
    
    @MCPMethod(description = "上传文档到知识库")
    public CompletableFuture<Map<String, Object>> uploadDocument(
            MultipartFile file, 
            String category, 
            Map<String, String> metadata) {
        
        return CompletableFuture.supplyAsync(() -> {
            try {
                // 1. 文档预处理
                Document document = documentProcessor.processDocument(file, category, metadata);
                
                // 2. 向量化处理
                List<Float> embedding = documentProcessor.generateEmbedding(document.getContent());
                
                // 3. 存储到向量库
                String docId = vectorStore.storeDocument(document, embedding);
                
                // 4. 更新索引
                searchIndex.updateIndex(docId, document);
                
                return Map.of(
                    "success", true,
                    "documentId", docId,
                    "message", "文档上传成功"
                );
            } catch (Exception e) {
                return Map.of(
                    "success", false,
                    "error", e.getMessage()
                );
            }
        });
    }
    
    @MCPMethod(description = "搜索知识库文档")
    public Map<String, Object> searchDocuments(String query, int topK, String category) {
        try {
            // 1. 生成查询向量
            List<Float> queryEmbedding = documentProcessor.generateEmbedding(query);
            
            // 2. 向量检索
            List<SearchResult> results = vectorStore.search(queryEmbedding, topK, category);
            
            // 3. 重排序
            List<SearchResult> rerankedResults = searchIndex.rerank(query, results);
            
            return Map.of(
                "success", true,
                "results", rerankedResults,
                "total", rerankedResults.size()
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
    
    @MCPMethod(description = "获取知识库统计信息")
    public Map<String, Object> getKnowledgeBaseStats() {
        try {
            long totalDocuments = vectorStore.getDocumentCount();
            long totalCategories = vectorStore.getCategoryCount();
            long totalVectors = vectorStore.getVectorCount();
            
            return Map.of(
                "success", true,
                "totalDocuments", totalDocuments,
                "totalCategories", totalCategories,
                "totalVectors", totalVectors,
                "lastUpdated", System.currentTimeMillis()
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
}
```

### 2. 智能问答服务

```java
// backend/src/main/java/com/example/ai/service/QuestionAnswerService.java
package com.example.ai.service;

import com.unionhole.mcp.annotation.MCPService;
import com.unionhole.mcp.annotation.MCPMethod;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

@Service
@MCPService(packageName = "com.example.ai.mcp")
public class QuestionAnswerService {
    
    @Autowired
    private RAGEngine ragEngine;
    
    @Autowired
    private AnswerGenerator answerGenerator;
    
    @Autowired
    private AnswerEvaluator answerEvaluator;
    
    @MCPMethod(description = "智能问答")
    public CompletableFuture<Map<String, Object>> askQuestion(
            String question, 
            String userId, 
            Map<String, Object> context) {
        
        return CompletableFuture.supplyAsync(() -> {
            try {
                // 1. 检索相关文档
                List<Document> relevantDocs = ragEngine.retrieveDocuments(question, 5);
                
                // 2. 生成答案
                String answer = answerGenerator.generateAnswer(question, relevantDocs, context);
                
                // 3. 评估答案质量
                AnswerEvaluation evaluation = answerEvaluator.evaluateAnswer(
                    question, answer, relevantDocs);
                
                // 4. 记录问答历史
                qaHistoryService.recordQA(userId, question, answer, evaluation);
                
                return Map.of(
                    "success", true,
                    "question", question,
                    "answer", answer,
                    "confidence", evaluation.getConfidence(),
                    "sources", relevantDocs,
                    "evaluation", evaluation
                );
            } catch (Exception e) {
                return Map.of(
                    "success", false,
                    "error", e.getMessage()
                );
            }
        });
    }
    
    @MCPMethod(description = "批量问答")
    public Map<String, Object> batchAskQuestions(
            List<String> questions, 
            String userId) {
        
        try {
            List<CompletableFuture<Map<String, Object>>> futures = questions.stream()
                .map(question -> askQuestion(question, userId, Map.of()))
                .toList();
            
            List<Map<String, Object>> results = futures.stream()
                .map(CompletableFuture::join)
                .toList();
            
            return Map.of(
                "success", true,
                "results", results,
                "total", results.size()
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
    
    @MCPMethod(description = "获取问答历史")
    public Map<String, Object> getQAHistory(String userId, int page, int size) {
        try {
            Page<QAHistory> history = qaHistoryService.getUserHistory(userId, page, size);
            
            return Map.of(
                "success", true,
                "history", history.getContent(),
                "totalPages", history.getTotalPages(),
                "totalElements", history.getTotalElements()
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
}
```

### 3. 安全防护服务

```java
// backend/src/main/java/com/example/ai/service/SecurityService.java
package com.example.ai.service;

import com.unionhole.mcp.annotation.MCPService;
import com.unionhole.mcp.annotation.MCPMethod;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.Map;

@Service
@MCPService(packageName = "com.example.ai.mcp")
public class SecurityService {
    
    @Autowired
    private InputDetector inputDetector;
    
    @Autowired
    private AdversarialDetector adversarialDetector;
    
    @Autowired
    private OutputMonitor outputMonitor;
    
    @Autowired
    private SecurityResponder securityResponder;
    
    @MCPMethod(description = "检测输入安全性")
    public Map<String, Object> detectInputSecurity(String input, String clientIp) {
        try {
            // 1. 输入检测
            InputDetectionResult inputResult = inputDetector.detectMaliciousInput(input);
            
            // 2. 对抗性攻击检测
            AdversarialDetectionResult adversarialResult = null;
            if (inputResult.isSuspicious()) {
                adversarialResult = adversarialDetector.detectAdversarialAttack(input);
            }
            
            // 3. 综合威胁评估
            ThreatAssessment threat = ThreatAssessment.builder()
                .inputDetection(inputResult)
                .adversarialDetection(adversarialResult)
                .clientIp(clientIp)
                .build();
            
            // 4. 安全响应
            SecurityResponse response = securityResponder.respondToThreat(threat);
            
            return Map.of(
                "success", true,
                "isSafe", response.isAllowed(),
                "threatLevel", threat.getLevel(),
                "actions", response.getActions(),
                "message", response.getMessage()
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
    
    @MCPMethod(description = "监控输出安全性")
    public Map<String, Object> monitorOutputSecurity(
            String output, 
            String question, 
            double confidence) {
        
        try {
            // 1. 输出监控
            OutputMonitoringResult monitoring = outputMonitor.monitorOutput(
                output, confidence, System.currentTimeMillis());
            
            // 2. 安全检查
            boolean isSafe = monitoring.getAnomalyScore() < 0.7;
            
            // 3. 记录安全事件
            if (!isSafe) {
                securityEventService.recordSecurityEvent(
                    SecurityEvent.builder()
                        .type("output_anomaly")
                        .severity("medium")
                        .details(monitoring)
                        .build());
            }
            
            return Map.of(
                "success", true,
                "isSafe", isSafe,
                "anomalyScore", monitoring.getAnomalyScore(),
                "recommendations", monitoring.getRecommendations()
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
    
    @MCPMethod(description = "获取安全统计信息")
    public Map<String, Object> getSecurityStats() {
        try {
            SecurityStats stats = securityService.getSecurityStats();
            
            return Map.of(
                "success", true,
                "totalRequests", stats.getTotalRequests(),
                "blockedRequests", stats.getBlockedRequests(),
                "threatsDetected", stats.getThreatsDetected(),
                "avgResponseTime", stats.getAvgResponseTime(),
                "lastUpdated", stats.getLastUpdated()
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
}
```

### 4. 系统监控服务

```java
// backend/src/main/java/com/example/ai/service/MonitoringService.java
package com.example.ai.service;

import com.unionhole.mcp.annotation.MCPService;
import com.unionhole.mcp.annotation.MCPMethod;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.Map;

@Service
@MCPService(packageName = "com.example.ai.mcp")
public class MonitoringService {
    
    @Autowired
    private SystemMetricsCollector metricsCollector;
    
    @Autowired
    private AlertManager alertManager;
    
    @MCPMethod(description = "获取系统健康状态")
    public Map<String, Object> getSystemHealth() {
        try {
            SystemHealth health = metricsCollector.getSystemHealth();
            
            return Map.of(
                "success", true,
                "status", health.getStatus(),
                "cpuUsage", health.getCpuUsage(),
                "memoryUsage", health.getMemoryUsage(),
                "diskUsage", health.getDiskUsage(),
                "activeConnections", health.getActiveConnections(),
                "lastChecked", health.getLastChecked()
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
    
    @MCPMethod(description = "获取性能指标")
    public Map<String, Object> getPerformanceMetrics(String timeRange) {
        try {
            PerformanceMetrics metrics = metricsCollector.getPerformanceMetrics(timeRange);
            
            return Map.of(
                "success", true,
                "avgResponseTime", metrics.getAvgResponseTime(),
                "throughput", metrics.getThroughput(),
                "errorRate", metrics.getErrorRate(),
                "successRate", metrics.getSuccessRate(),
                "timeRange", timeRange
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
    
    @MCPMethod(description = "获取告警信息")
    public Map<String, Object> getAlerts(String severity) {
        try {
            List<Alert> alerts = alertManager.getAlerts(severity);
            
            return Map.of(
                "success", true,
                "alerts", alerts,
                "total", alerts.size(),
                "severity", severity
            );
        } catch (Exception e) {
            return Map.of(
                "success", false,
                "error", e.getMessage()
            );
        }
    }
}
```

### 5. 前端React组件

```jsx
// frontend/src/components/QuestionAnswer.jsx
import React, { useState, useEffect } from 'react';
import { Card, Input, Button, List, Typography, Spin, Alert } from 'antd';
import { SendOutlined, HistoryOutlined } from '@ant-design/icons';

const { TextArea } = Input;
const { Title, Paragraph } = Typography;

const QuestionAnswer = () => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState('');
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  const askQuestion = async () => {
    if (!question.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/mcp/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          userId: 'current-user',
          context: {}
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setAnswer(data.answer);
        setHistory(prev => [{
          question: data.question,
          answer: data.answer,
          confidence: data.confidence,
          timestamp: new Date().toISOString()
        }, ...prev]);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('网络错误，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    try {
      const response = await fetch('/api/mcp/history?userId=current-user&page=0&size=10');
      const data = await response.json();
      
      if (data.success) {
        setHistory(data.history);
      }
    } catch (err) {
      console.error('加载历史失败:', err);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <Card title="智能问答系统" style={{ marginBottom: '24px' }}>
        <div style={{ marginBottom: '16px' }}>
          <TextArea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="请输入您的问题..."
            rows={4}
            style={{ marginBottom: '12px' }}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={askQuestion}
            loading={loading}
            disabled={!question.trim()}
            style={{ width: '100%' }}
          >
            提问
          </Button>
        </div>
        
        {error && (
          <Alert
            message="错误"
            description={error}
            type="error"
            style={{ marginBottom: '16px' }}
          />
        )}
        
        {loading && (
          <div style={{ textAlign: 'center', padding: '24px' }}>
            <Spin size="large" />
            <div style={{ marginTop: '12px' }}>正在思考中...</div>
          </div>
        )}
        
        {answer && !loading && (
          <Card title="回答" style={{ marginTop: '16px' }}>
            <Paragraph>{answer}</Paragraph>
          </Card>
        )}
      </Card>
      
      <Card
        title={
          <span>
            <HistoryOutlined style={{ marginRight: '8px' }} />
            问答历史
          </span>
        }
      >
        <List
          dataSource={history}
          renderItem={(item, index) => (
            <List.Item key={index}>
              <div style={{ width: '100%' }}>
                <Title level={5}>问题: {item.question}</Title>
                <Paragraph>回答: {item.answer}</Paragraph>
                <Text type="secondary">
                  置信度: {(item.confidence * 100).toFixed(1)}% | 
                  时间: {new Date(item.timestamp).toLocaleString()}
                </Text>
              </div>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default QuestionAnswer;
```

### 6. Docker部署配置

```yaml
# deploy/docker/docker-compose.yml
version: '3.8'

services:
  ai-system-backend:
    build: 
      context: ../../backend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=prod
      - DATABASE_URL=jdbc:postgresql://postgres:5432/ai_system
      - REDIS_URL=redis://redis:6379
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on:
      - postgres
      - redis
      - elasticsearch
    volumes:
      - ./logs:/app/logs
    networks:
      - ai-network

  ai-system-frontend:
    build:
      context: ../../frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8080
    depends_on:
      - ai-system-backend
    networks:
      - ai-network

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ai_system
      - POSTGRES_USER=ai_user
      - POSTGRES_PASSWORD=ai_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - ai-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ai-network

  elasticsearch:
    image: elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    networks:
      - ai-network

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - ai-network

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - ai-network

volumes:
  postgres_data:
  redis_data:
  elasticsearch_data:
  grafana_data:

networks:
  ai-network:
    driver: bridge
```

## 测试验证

### 1. 单元测试

```java
// backend/src/test/java/com/example/ai/service/QuestionAnswerServiceTest.java
@SpringBootTest
class QuestionAnswerServiceTest {
    
    @Autowired
    private QuestionAnswerService questionAnswerService;
    
    @Test
    void testAskQuestion() {
        String question = "什么是人工智能？";
        String userId = "test-user";
        Map<String, Object> context = Map.of();
        
        CompletableFuture<Map<String, Object>> future = 
            questionAnswerService.askQuestion(question, userId, context);
        
        Map<String, Object> result = future.join();
        
        assertTrue((Boolean) result.get("success"));
        assertNotNull(result.get("answer"));
        assertTrue((Double) result.get("confidence") > 0.5);
    }
}
```

### 2. 集成测试

```bash
# 启动系统
docker-compose up -d

# 测试API
curl -X POST http://localhost:8080/api/mcp/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是机器学习？", "userId": "test-user"}'

# 测试安全防护
curl -X POST http://localhost:8080/api/mcp/security/detect \
  -H "Content-Type: application/json" \
  -d '{"input": "正常问题", "clientIp": "192.168.1.100"}'
```

### 3. 性能测试

```bash
# 使用Apache Bench进行压力测试
ab -n 1000 -c 10 -H "Content-Type: application/json" \
  -p test_data.json http://localhost:8080/api/mcp/ask
```

## 核心能力总结

通过本综合实战项目，我们展示了以下核心能力：

1. **系统架构设计**: 微服务架构、分层设计、模块化开发
2. **技术栈整合**: Spring Boot、React、PostgreSQL、Redis、Elasticsearch
3. **AI能力集成**: MCP协议、RAG技术、安全防护
4. **工程化实践**: Docker容器化、监控告警、日志管理
5. **用户体验**: 响应式前端、实时交互、错误处理

## 扩展方向

1. **多模态支持**: 支持图像、音频等多种输入类型
2. **分布式部署**: Kubernetes集群部署、服务网格
3. **智能优化**: 自动调参、模型热更新、A/B测试
4. **企业集成**: SSO认证、LDAP集成、审计日志
5. **国际化**: 多语言支持、本地化部署

## 学习要点

- 掌握大型AI系统的架构设计
- 学习微服务开发和部署实践
- 了解AI技术的工程化应用
- 实践DevOps和监控运维
- 培养全栈开发能力

---

**通过这个综合实战项目，你将掌握企业级AI系统开发的核心技能！**
