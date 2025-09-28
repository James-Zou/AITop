# MCP协议快速入门实战

## 项目概述

基于 [mcp-springboot-server](https://github.com/James-Zou/mcp-springboot-server) 和 [mcp-facade-generator](https://github.com/James-Zou/mcp-facade-generator) 的核心能力，快速构建一个支持MCP协议的AI服务。

## 核心价值分析

### mcp-springboot-server 核心实现
- **自动工具注册**: 通过`McpServerConfig`类自动扫描和注册MCP工具
- **Spring AI集成**: 无缝集成Spring AI框架，简化AI服务开发
- **RESTful API支持**: 提供标准的HTTP接口和SSE支持
- **配置驱动**: 通过配置文件快速定制服务行为

### mcp-facade-generator 核心实现
- **代码自动生成**: 基于注解自动生成MCP协议处理类
- **方法级控制**: 通过`@MCPMethod`注解精确控制生成内容
- **异常处理优化**: 统一的异常处理和错误响应机制
- **Demo项目生成**: 一键生成完整的示例项目

## 实战目标

构建一个智能天气查询服务，展示MCP协议的核心能力：
1. 自动生成MCP Facade类
2. 实现工具自动注册
3. 提供RESTful API接口
4. 支持实时数据推送

## 环境准备

### 1. 项目初始化

```bash
# 创建Spring Boot项目
mvn archetype:generate \
  -DgroupId=com.example \
  -DartifactId=weather-mcp-service \
  -DarchetypeArtifactId=maven-archetype-quickstart \
  -DinteractiveMode=false

cd weather-mcp-service
```

### 2. 添加依赖

```xml
<dependencies>
    <!-- Spring Boot Starter -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
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
</dependencies>
```

### 3. 配置Maven编译器

```xml
<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.11.0</version>
            <configuration>
                <source>17</source>
                <target>17</target>
                <annotationProcessorPaths>
                    <path>
                        <groupId>com.unionhole</groupId>
                        <artifactId>mcp-facade-generator</artifactId>
                        <version>1.0.1</version>
                    </path>
                </annotationProcessorPaths>
                <compilerArgs>
                    <arg>-Amcp.demo.output=true</arg>
                </compilerArgs>
            </configuration>
        </plugin>
    </plugins>
</build>
```

## 核心实现

### 1. 业务服务类

```java
package com.example.weather.service;

import com.unionhole.mcp.annotation.MCPService;
import com.unionhole.mcp.annotation.MCPMethod;
import org.springframework.stereotype.Service;
import java.util.Map;
import java.util.HashMap;

@Service
@MCPService(packageName = "com.example.weather.mcp")
public class WeatherService {
    
    @MCPMethod(description = "获取指定城市的天气信息")
    public Map<String, Object> getWeather(String cityName) {
        // 模拟天气数据
        Map<String, Object> weather = new HashMap<>();
        weather.put("city", cityName);
        weather.put("temperature", "22°C");
        weather.put("condition", "晴天");
        weather.put("humidity", "65%");
        weather.put("windSpeed", "5 km/h");
        return weather;
    }
    
    @MCPMethod(description = "获取未来7天天气预报")
    public Map<String, Object> getForecast(String cityName) {
        Map<String, Object> forecast = new HashMap<>();
        forecast.put("city", cityName);
        forecast.put("forecast", new String[]{
            "周一: 晴天, 22°C",
            "周二: 多云, 20°C", 
            "周三: 小雨, 18°C",
            "周四: 晴天, 24°C",
            "周五: 多云, 21°C",
            "周六: 晴天, 25°C",
            "周日: 晴天, 23°C"
        });
        return forecast;
    }
    
    @MCPMethod(description = "获取天气预警信息")
    public Map<String, Object> getWeatherAlert(String cityName) {
        Map<String, Object> alert = new HashMap<>();
        alert.put("city", cityName);
        alert.put("hasAlert", false);
        alert.put("message", "当前无天气预警");
        return alert;
    }
}
```

### 2. MCP服务器配置

```java
package com.example.weather.config;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.ai.mcp.server.McpServer;
import org.springframework.ai.mcp.server.McpServerConfig;

@Configuration
public class McpServerConfiguration {
    
    @Bean
    public McpServerConfig mcpServerConfig() {
        return McpServerConfig.builder()
            .name("weather-service")
            .version("1.0.0")
            .description("智能天气查询服务")
            .build();
    }
    
    @Bean
    public McpServer mcpServer() {
        return new McpServer(mcpServerConfig());
    }
}
```

### 3. 应用配置

```properties
# application.properties
spring.application.name=weather-mcp-service
server.port=8080

# MCP配置
mcp.server.name=weather-service
mcp.server.version=1.0.0
mcp.server.description=智能天气查询服务

# 日志配置
logging.level.com.example.weather=DEBUG
logging.level.org.springframework.ai=DEBUG
```

### 4. 主应用类

```java
package com.example.weather;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication
@ComponentScan(basePackages = {"com.example.weather", "com.unionhole.mcp"})
public class WeatherMcpServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(WeatherMcpServiceApplication.class, args);
    }
}
```

## 自动生成代码

编译项目后，mcp-facade-generator会自动生成以下文件：

### 生成的Facade类

```java
package com.example.weather.mcp;

import com.unionhole.mcp.vo.MCPRequest;
import com.unionhole.mcp.vo.MCPResponse;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.stereotype.Component;
import com.example.weather.service.WeatherService;

@Component
public class WeatherServiceFacade {
    private final WeatherService service;

    public WeatherServiceFacade(WeatherService service) {
        this.service = service;
    }

    @Tool(description = "获取指定城市的天气信息")
    public MCPResponse getWeather(MCPRequest request) {
        try {
            String cityName = request.getParameter("cityName", String.class);
            Object result = service.getWeather(cityName);
            return MCPResponse.success(result);
        } catch (Exception e) {
            return MCPResponse.error(e.getMessage());
        }
    }

    @Tool(description = "获取未来7天天气预报")
    public MCPResponse getForecast(MCPRequest request) {
        try {
            String cityName = request.getParameter("cityName", String.class);
            Object result = service.getForecast(cityName);
            return MCPResponse.success(result);
        } catch (Exception e) {
            return MCPResponse.error(e.getMessage());
        }
    }

    @Tool(description = "获取天气预警信息")
    public MCPResponse getWeatherAlert(MCPRequest request) {
        try {
            String cityName = request.getParameter("cityName", String.class);
            Object result = service.getWeatherAlert(cityName);
            return MCPResponse.success(result);
        } catch (Exception e) {
            return MCPResponse.error(e.getMessage());
        }
    }
}
```

## 测试验证

### 1. 启动服务

```bash
mvn spring-boot:run
```

### 2. 测试MCP工具

```bash
# 测试天气查询
curl -X POST http://localhost:8080/mcp/tools/weather \
  -H "Content-Type: application/json" \
  -d '{"cityName": "北京"}'

# 测试天气预报
curl -X POST http://localhost:8080/mcp/tools/forecast \
  -H "Content-Type: application/json" \
  -d '{"cityName": "上海"}'

# 测试天气预警
curl -X POST http://localhost:8080/mcp/tools/alert \
  -H "Content-Type: application/json" \
  -d '{"cityName": "广州"}'
```

### 3. 查看MCP工具列表

```bash
curl http://localhost:8080/mcp/tools
```

## 核心能力总结

通过本实战项目，我们展示了以下核心能力：

1. **代码自动生成**: 通过注解驱动，自动生成MCP协议处理类
2. **工具自动注册**: Spring容器自动发现和注册MCP工具
3. **标准化接口**: 提供统一的MCP协议接口
4. **异常处理**: 统一的错误处理和响应机制
5. **配置驱动**: 通过配置文件灵活定制服务行为

## 扩展方向

1. **集成真实天气API**: 替换模拟数据为真实天气服务
2. **添加缓存机制**: 提高查询性能
3. **实现SSE推送**: 支持实时天气更新
4. **添加认证授权**: 保护API安全
5. **监控和日志**: 添加完整的监控体系

## 学习要点

- 理解MCP协议的核心概念
- 掌握注解驱动的代码生成
- 学习Spring AI框架的集成方式
- 了解微服务架构的设计模式
- 实践RESTful API的设计原则

---

**通过这个实战项目，你将掌握MCP协议开发的核心技能！**
