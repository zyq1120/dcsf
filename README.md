# 智能文档识别与信息提取系统（Flask）

> 面向高校常见文档（请/销假单、学籍卡、成绩单、奖助、助学贷款合同等）的 OCR + NLP +（可选）LLM 结构化抽取服务。

- Linux 部署指南（含 CPU/内存建议、systemd/gunicorn/Docker）：[`DEPLOY_LINUX.md`](./DEPLOY_LINUX.md)

## 快速开始（本地）

### 1) 创建虚拟环境

本项目在不同机器上可能存在 `venv/` 与 `.venv/` 两种虚拟环境目录。

- 推荐统一使用 `venv/`（README 默认按 `venv` 给命令）。
- 如果你已有 `.venv/`，请确保 `.venv/pyvenv.cfg` 存在；否则该环境会报 `No pyvenv.cfg file`，无法运行。

Windows（PowerShell）：

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Windows（cmd.exe）：

```bat
python -m venv venv
venv\Scripts\activate.bat
```

### 2) 安装依赖

```bat
pip install -r requirements.txt
```

### 3) 启动服务

```bat
python app.py
```

默认地址：`http://127.0.0.1:5000`

### 4) 预热（可选但推荐）

项目支持启动期预热（加载 NLP / LLM / 简洁版 PaddleOCR 引擎），减少首请求延迟。

- 代码入口：`app/services/factory.py -> preload_all_services()`
- 开关：`WARMUP_SIMPLE_OCR`（默认 true）

> 注意：预热 PaddleOCR 会占用一段 CPU/内存（加载模型），但能显著减少首请求卡顿。

---

## 1. 引言

### 1.1 项目背景

在高校日常管理和学生事务处理中，涉及大量纸质或电子文档，如成绩单、在校证明、助学贷款合同、请假条等。传统的人工审核与信息录入方式效率低下、易出错，且难以适应日益增长的数字化需求。为了解决这一痛点，本项目旨在设计并实现一个智能文档处理系统，利用人工智能技术自动完成文档的分类、文字识别（OCR）、关键信息提取（IE）和结构化输出。

### 1.2 系统目标

本系统致力于提供一个统一、高效、智能的后端服务，能够：
- **自动化处理**：接收图片或 PDF 格式的文档，自动进行分析和信息提取。
- **高精度识别**：结合先进的 OCR、NLP 和大语言模型（LLM）技术，确保信息提取的准确性。
- **结构化输出**：将非结构化的文档内容转化为统一、标准的 JSON 格式，便于上层业务系统（如 Spring Boot 管理平台）集成和使用。
- **高可扩展性**：系统采用模块化设计，易于扩展以支持更多文档类型和新的 AI 模型。

## 2. 系统设计

### 2.1 系统架构

本系统作为核心 AI 后端，被设计为由上层业务系统（如基于 Spring Boot 的 Web 应用）通过 HTTP API 进行调用。整体架构遵循分层设计思想，确保各模块职责清晰、易于维护。

```mermaid
graph TD
    subgraph "用户端"
        User[<fa:fa-user> 用户]
    end

    subgraph "上层业务系统 (Spring Boot)"
        WebApp[<fa:fa-window-maximize> Web 应用 / API 网关]
    end

    subgraph "智能文档处理系统 (Flask)"
        APILayer[<fa:fa-server> API 层 (Flask)]
        Orchestrator[<fa:fa-cogs> FinalAIService 编排服务]
        
        subgraph "核心 AI 服务"
            OCR[<fa:fa-camera> OCR 服务 (PaddleOCR)]
            NLP[<fa:fa-language> NLP 服务 (Regex/Spacy 可选)]
            LLM[<fa:fa-robot> LLM 服务 (Qwen/Gemini/OpenAI 可选)]
        end

        subgraph "支撑模块"
            Cache[<fa:fa-database> 缓存 (JSON)]
            Logger[<fa:fa-file-alt> 日志 (Loguru)]
            Config[<fa:fa-cog> 配置管理]
        end
    end

    User --> WebApp
    WebApp -- "HTTP API 调用 (POST /api/v1/ai/process)" --> APILayer
    APILayer --> Orchestrator
    Orchestrator --> OCR
    Orchestrator --> NLP
    Orchestrator --> LLM
    Orchestrator --> Cache
    Orchestrator --> Logger
    Orchestrator --> Config

    style WebApp fill:#f9f,stroke:#333,stroke-width:2px
    style APILayer fill:#bbf,stroke:#333,stroke-width:2px
```

### 2.2 OCR 设计说明（简洁模式优先）

系统同时提供两套 OCR：

- **简洁版 `PaddleOCRService`**（默认路径、与 `tests/test_ocr.py` 逻辑对齐）
  - 直接调用 `PaddleOCR.ocr()`
  - 日志会打印识别文本内容
  - 针对 Windows 上偶发 `RuntimeError: could not execute a primitive` 已做**自动重试/降级**：
    - 首次正常 cls=True
    - 若命中 primitive 错误：短延迟后重试一次，并降级 cls=False

- **复杂版 `OCRService`**（表格/强制复杂/图像增强/手写模式等）
  - 默认不在启动期加载重引擎（lazy init），避免 CPU/内存占用过高
  - 当启用 `detect_table/table_force/use_complex_mode/handwriting_mode/enhance_image` 等选项时触发

> 说明：即使走简洁 OCR，如果遇到 primitive 类偶发错误，`OCRService` 也会自动回退到“预处理 + 内存识别”再试一次，尽量避免请求直接 500。

## 3. 系统实现

### 3.1 项目结构

```
.
├── app.py                      # 应用入口
├── app/
│   ├── __init__.py             # create_app
│   ├── api/
│   │   ├── ai.py               # /api/v1/ai/process
│   │   └── health.py           # /health, /api/v1/health...
│   ├── services/
│   │   ├── final_ai_service.py # 编排服务
│   │   ├── ocr_service.py      # OCR 统一入口（默认简洁，复杂按需）
│   │   ├── paddle_ocr_service.py # 简洁 OCR（对齐 test_ocr 逻辑）
│   │   ├── nlp_service.py      # NLP 分类/实体/字段
│   │   ├── llm_service.py      # LLM（可选）
│   │   └── factory.py          # 单例工厂 + preload
│   ├── templates/              # 内置模板
│   └── utils/
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### 3.2 核心处理流程

系统处理文档的核心流程由 `FinalAIService` 编排，具体步骤如下图所示：

```mermaid
graph TD
    A[接收请求 /api/v1/ai/process] --> B{输入类型判断};
    B -- "文件 (图片/PDF)" --> C[OCR 识别];
    B -- "纯文本" --> D[直接进入 NLP];
    B -- "文件 + 多模态选项" --> E[LLM 多模态直接处理];
    
    C --> F{OCR 结果有效?};
    F -- "是" --> G[LLM 增强 OCR (可选)];
    F -- "否" --> H[触发 LLM 多模态兜底];
    
    G --> D;
    H -- "成功" --> E;
    H -- "失败" --> I[返回错误];

    D --> J[NLP 处理: 文档分类、实体识别、规则字段提取];
    J --> K{NLP 提取成功?};
    K -- "是" --> L[合并字段];
    K -- "否 / 结果不佳" --> M[触发 LLM 大语言模型兜底];
    
    M -- "成功" --> N[LLM 提取字段];
    M -- "失败" --> L;
    N --> L;

    E --> O[LLM 直接输出结构化 JSON];
    O --> L;

    L --> P[结果整合与后处理];
    P --> Q[格式化为统一 JSON 结构];
    Q --> R[返回响应];

    style E fill:#d4edda,stroke:#155724
    style H fill:#f8d7da,stroke:#721c24
    style M fill:#fff3cd,stroke:#856404
```

该流程确保了系统在不同场景下的灵活性和鲁棒性。例如，对于标准格式的成绩单，可以通过高效的 OCR+NLP 规则快速完成；而对于格式复杂的合同，则可以利用 LLM 的强大理解能力进行兜底处理。

## 4. API

### 4.1 健康检查

| 方法 | 路径 | 功能描述 |
| :--- | :--- | :--- |
| `GET` | `/health` | 通用健康检查 |
| `GET` | `/api/v1/health` | V1 健康检查 |
| `GET` | `/api/v1/ocr/health` | OCR 健康检查（会返回 simple_ocr_ready） |
| `GET` | `/api/v1/nlp/health` | NLP 健康检查 |

### 4.2 `/api/v1/ai/process`

#### 4.2.1 Body

| 字段名 | 类型 | 是否必须 | 描述 |
| :--- | :--- | :--- | :--- |
| `file_content` | string | 否 | 文件 Base64（推荐：dataURL 或纯 base64）。与 `text` 至少提供一个。 |
| `file_name` | string | 否 | 原始文件名（用于识别 .pdf 等）。 |
| `text` | string | 否 | 纯文本模式（跳过 OCR）。 |
| `options` | object | 否 | 流程控制参数。 |
| `template_config` | object | 否 | 自定义模板（fields/required_fields 等）。 |

#### 4.2.2 options（实际实现对齐）

| 选项 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| `llm_only` | boolean | false | 仅走 LLM（跳过 OCR/NLP）。 |
| `disable_ai` | boolean | false | 禁用 LLM（包括兜底与增强）。 |
| `llm_image` | boolean | false | 允许多模态 LLM 直读图片（可绕过 OCR）。 |
| `auto_infer_fields` | boolean | false | 自动推断字段并生成临时模板。 |
| `analyze_ocr` | boolean | false | 生成 OCR 诊断（类型/质量/KV/表格重建等）。 |
| `detect_table` | boolean | false | 表格检测（复杂 OCR 路径）。 |
| `table_force` | boolean | false | 强制表格 OCR（复杂 OCR 路径）。 |
| `enhance_image` | boolean | true | 图像增强（开启则会走预处理/内存识别）。 |
| `handwriting_mode` | boolean | false | 手写模式（会触发预处理链）。 |
| `llm_provider` | string | null | 覆盖 LLM provider。 |
| `llm_model` | string | null | 覆盖 LLM model。 |
| `table_detect_threshold` | float | 0.002 | 表格 heuristics 阈值。 |

> 说明：历史文档中的 `use_llm` 控制项在当前代码中不作为强制开关，LLM 是否可用主要由 `disable_ai` 与服务端 LLM 配置决定。

## 5. 开发与测试

### 5.1 运行单测

Windows（cmd.exe）：

```bat
venv\Scripts\python.exe -m pytest -q -k "not test_ocr"
```

> 如果你机器上 PaddleOCR 模型较大、或未准备好 OCR 环境，建议先跳过 `test_ocr`。

## 6. 部署

本项目支持 Docker / docker-compose 部署（详见 `Dockerfile` / `docker-compose.yml`）。

---

## 7. 常见问题（FAQ）

### Q1：为什么偶尔出现 `RuntimeError: could not execute a primitive`？

这是 Paddle 推理底层（predictor.run / oneDNN / MKL-DNN）在部分 Windows/CPU 环境下的偶发问题，常见诱因包括线程调度、瞬时内存、以及 PDF 转图片后文件尚未完全稳定等。

当前项目已做两层容错：
- `PaddleOCRService`：遇到 primitive 错误自动重试并降级（cls=False）
- `OCRService`：简洁 OCR 失败时自动回退到“预处理 + 内存识别”再试一次

---

## 8. 结论与展望

本项目成功实现了一个功能全面、性能可靠的智能文档处理系统。通过结合传统 OCR/NLP 技术与前沿的大语言模型，系统在保证处理效率的同时，也具备了处理复杂和非标文档的能力。

未来工作可从以下几个方面展开：
- OCR/NLP 规则持续扩展（覆盖更多高校常见文档）
- 性能提升：异步队列（Celery/RQ）、限流与缓存
- 文档知识库 + RAG 提升语义抽取/纠错能力
