# Linux 部署文档（建议配置与调优）

本文档面向 **Linux 生产环境**（Ubuntu / Debian / CentOS / Rocky 等），用于部署本项目：`document_classification_system_flask`。

目标：
- 部署可用（HTTP API 正常响应）
- OCR 首次加载稳定、避免频繁 500
- 给出明确的 CPU / 内存建议（包含但不限于）

---

## 1. 推荐硬件规格（按场景分档）

> 说明：本服务的主要资源消耗来自 **PaddleOCR 推理** 与 **PDF->图片转换**。
> 如果启用 LLM（尤其多模态），还会增加网络等待与少量 CPU（序列化/图片 base64/JSON 处理），但主要瓶颈仍然是 OCR。

### A. 低并发 / 开发测试

- CPU：**2 vCPU**（或 2 core）
- 内存：**4 GB**
- 适用：单人调试、低频调用（每分钟 < 10 次）

### B. 中等并发 / 小团队生产

- CPU：**4~8 vCPU**
- 内存：**8~16 GB**
- 适用：多用户并发、日志持续写入、PDF/图片混合请求

### C. 高并发 / 生产稳定（推荐）

- CPU：**8~16 vCPU**
- 内存：**16~32 GB**
- 适用：并发 OCR、批量处理、较多 PDF

### D. 说明：为什么需要这些资源？

- **PaddleOCR 模型加载**：首次加载会占用明显 CPU/内存，并产生模型文件缓存（`~/.paddleocr`）。
- **PDF 转图片（pdf2image + poppler）**：对大 PDF 会短时占用较多 CPU 与内存。
- **并发请求**：如果你开多进程/多线程，推理会放大资源占用。建议通过 gunicorn 进程数控制并发。

---

## 2. Linux 依赖准备（非 Docker 部署）

### 2.1 系统包

以 Ubuntu/Debian 为例：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  poppler-utils \
  libgl1 \
  libglib2.0-0
```

说明：
- `poppler-utils`：用于 `pdf2image` 转换 PDF
- `libgl1`、`libglib2.0-0`：OpenCV 常见运行依赖

> CentOS/Rocky 可用 `yum/dnf` 安装对应包（poppler-utils / mesa-libGL / glib2）。

### 2.2 Python 版本

建议：**Python 3.10**（与项目依赖最匹配）。

---

## 3. 代码部署（非 Docker）

### 3.1 拉取代码

```bash
cd /opt
git clone <your-repo-url> document_classification_system_flask
cd document_classification_system_flask
```


### 3.2 安装依赖（两种方式）

方式 A（推荐）：使用 `venv`

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

方式 B（可选）：不使用虚拟环境（仅建议开发机）

```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

> 说明：生产环境建议优先使用方式 A，避免系统 Python 被污染或与其他服务发生依赖冲突。

> 注意：PaddlePaddle / PaddleOCR 在不同平台需要匹配版本。以 `requirements.txt` 为准。

### 3.3 配置环境变量

建议用 `.env` 或 systemd EnvironmentFile。关键项：

- LLM（可选）
- OCR（是否启用 GPU、是否预热）

示例（仅示意，按你项目实际 settings 为准）：

```bash
export WARMUP_SIMPLE_OCR=true
export USE_GPU=false
```

### 3.4 启动（开发模式，不推荐生产）

若使用方式 A（已激活 `venv`）：

```bash
python app.py
```

若使用方式 B（不使用 `venv`）：

```bash
python3 app.py
```

---

## 4. 生产部署（推荐：gunicorn + systemd）

> 建议：生产环境默认使用虚拟环境（`venv`），以下命令按该前提给出。

### 4.1 gunicorn 安装

```bash
source venv/bin/activate
python -m pip install gunicorn
```

### 4.2 进程/线程建议（非常重要）

PaddleOCR 在部分环境下可能会因为线程与底层算子选择导致偶发错误。

推荐从保守配置开始：

- `workers = CPU 核数 / 2`（取整，至少 1）
- `threads = 1`（优先保证稳定）

示例：8 vCPU 机器
- workers=4
- threads=1

启动示例：

```bash
source venv/bin/activate
gunicorn -w 4 -k gthread --threads 1 -b 0.0.0.0:5005 app:app
```

> 如果你使用 `create_app` 工厂模式，请改为对应入口（例如 `app:create_app()` 的 WSGI 包装）。

### 4.3 systemd 示例

创建环境变量文件（推荐）：`/opt/document_classification_system_flask/.env`：

```bash
PYTHONUNBUFFERED=1
WARMUP_SIMPLE_OCR=true
```

创建 `/etc/systemd/system/ai-doc.service`：

```ini
[Unit]
Description=AI Document Service (Flask)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/document_classification_system_flask

EnvironmentFile=/opt/document_classification_system_flask/.env
ExecStart=/opt/document_classification_system_flask/venv/bin/gunicorn -w 4 -k gthread --threads 1 -b 0.0.0.0:5005 app:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-doc
sudo systemctl start ai-doc
sudo systemctl status ai-doc
```

---

## 5. Docker 部署（推荐生产）

### 5.1 docker-compose 启动

```bash
docker-compose up -d
```

建议在 `docker-compose.yml` 中设置：
- 端口映射
- 日志卷
- `~/.paddleocr` 缓存卷（可选，但建议持久化）

示例（概念性说明，具体以你的 compose 为准）：
- 将容器内 `~/.paddleocr` 映射到宿主机目录，避免每次拉起都重下模型。

### 5.2 Docker 资源限制建议

- 低并发：`--cpus=2 --memory=4g`
- 中并发：`--cpus=4 --memory=8g`
- 高并发：`--cpus=8 --memory=16g`

---

## 6. 预热与启动优化

### 6.1 预热（强烈建议开启）

项目有预热入口：`app/services/factory.py -> preload_all_services()`。

建议：
- 生产：开启 `WARMUP_SIMPLE_OCR=true`
- 这样首个 OCR 请求不会卡很久，且可降低并发下首次加载导致的失败概率

### 6.2 PaddleOCR 偶发错误说明

如果你在日志中看到：

- `RuntimeError: could not execute a primitive`

这通常是底层推理引擎在特定 CPU/线程/内存瞬间条件下偶发失败。

当前代码已实现容错：
- 简洁 OCR：自动重试一次并降级（cls=False）
- OCRService：简洁 OCR 如果失败，会自动回退到“预处理 + 内存识别”再试一次

如果你仍然频繁遇到：
- 降低 gunicorn 并发（workers/threads）
- 确保机器内存充足
- 给 `~/.paddleocr` 做持久化，减少模型/缓存重复初始化

---

## 7. 监控与日志

- 日志目录：项目默认写入 `logs/`（以实际配置为准）
- 建议接入：
  - 进程监控：systemd / supervisor
  - 日志采集：ELK / Loki
  - 指标：Prometheus（可后续增加）

建议关注：
- 请求耗时（尤其 OCR）
- 500 错误比例
- CPU 使用率是否长期 100%
- 内存是否持续增长（可能需要重启策略/排查泄漏）

---

## 8. 快速自检

健康检查：

```bash
curl -s http://127.0.0.1:5005/health
curl -s http://127.0.0.1:5005/api/v1/health
curl -s http://127.0.0.1:5005/api/v1/ocr/health
```

核心接口：

- `POST /api/v1/ai/process`

建议先用小图片做 smoke test，再逐步上 PDF。
