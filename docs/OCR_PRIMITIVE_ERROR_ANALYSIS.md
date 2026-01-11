# OCR Primitive 执行失败问题分析与解决方案

## 问题现象（根据你的日志）

```
2025-12-26 14:34:14 | WARNING | OCR (Simple Mode) | primitive 执行失败，触发重试降级
2025-12-26 14:34:16 | ERROR   | OCR (Simple Mode) | 重试仍失败
2025-12-26 14:34:16 | WARNING | Simple OCR failed with primitive error, fallback to preprocessed memory OCR
2025-12-26 14:34:22 | WARNING | OCR (Simple Mode) | primitive 执行失败（内存图像），触发重试降级
2025-12-26 14:34:25 | ERROR   | OCR (Simple Mode) | 内存图像重试仍失败
2025-12-26 14:34:25 | ERROR   | OCR 识别失败: OCR 识别失败
127.0.0.1 - - [26/Dec/2025 14:34:26] "POST /api/v1/ai/process HTTP/1.1" 500 -
```

**关键特征：**
- 文件路径 OCR：第一次失败 → 重试失败
- 预处理 + 内存识别：也失败 → 重试失败
- **最终 500**

---

## 根本原因分析

### 1. 这不是"偶发"，而是"系统性失败"

你这个错误与之前的"偶发 primitive 错误"**性质不同**：

- **偶发型**：同一个文件，这次失败，下次成功（原因：线程调度/瞬时内存/oneDNN 算子选择不稳定）
- **系统性**：这个文件**每次都失败**，说明：
  - PDF 转出来的图片格式/尺寸/通道对当前 PaddlePaddle 推理引擎不兼容
  - 机器 CPU 指令集/oneDNN/MKL 版本与 PaddlePaddle wheel 不匹配
  - 内存严重不足（连续推理失败）

### 2. 为什么前两次重试都失败？

当前代码的重试策略：
- **第 1 次**：cls=True（正常模式）
- **第 2 次**：cls=False（降级，减少算子）
- **第 3 次**（新增）：降采样到 800px 短边 + cls=False

你的日志显示前两次都失败了，说明**不是算子选择的问题**，而是：
- 图片本身的解码/预处理在 Paddle 推理引擎中就炸了
- 或者机器资源（内存/CPU）已经到极限

---

## 已实施的解决方案（当前代码）

### A. PaddleOCRService 三次降级策略

我已经在 `paddle_ocr_service.py` 中加入**第三次终极兜底**：

#### recognize（文件路径）
```python
try:
    result = ocr(file_path, cls=True)  # 第 1 次
except primitive error:
    try:
        result = ocr(file_path, cls=False)  # 第 2 次
    except:
        try:
            result = _last_resort_ocr(file_path)  # 第 3 次：降采样 + 最简模式
        except:
            raise ServiceError("已尝试所有降级策略仍失败")
```

#### recognize_image（内存图像）
```python
try:
    result = ocr(image, cls=True)  # 第 1 次
except primitive error:
    try:
        result = ocr(image, cls=False)  # 第 2 次
    except:
        try:
            downsampled = _downsample_image(image)  # 降采样
            result = ocr(downsampled, cls=False)    # 第 3 次
        except:
            raise ServiceError("内存图像已尝试所有降级策略仍失败")
```

### B. 终极兜底方法：`_last_resort_ocr`

策略：
- 用 cv2 加载图片
- 强制缩小到短边 800px（极度降采样）
- 转为最简单的 BGR 格式（避免 RGBA/灰度兼容问题）
- 用 `cls=False` 调用 OCR

### C. OCRService 兜底（已存在）

`ocr_service.py` 在简洁 OCR 失败时，会自动回退到"预处理 + 内存识别"再试一次（你日志里看到的就是这个）。

---

## 为什么你的日志中仍然失败？

**可能原因（按概率排序）：**

### 1. 图片尺寸过大（最可能）

你的 PDF 转出来的图片可能是：
- 短边 > 2000px
- 长边 > 5000px
- 像素总数 > 1000 万

在 Windows + CPU 推理环境下，这种图片会导致：
- 内存占用暴增
- 推理引擎初始化 primitive 失败

**验证方法：**
在你日志中找 `PDF 已转换为图片` 那一行之后，加一行打印图片尺寸的日志。

### 2. PaddlePaddle 与 CPU 指令集不匹配

你的 CPU 可能不支持 AVX2/AVX512，而你安装的 PaddlePaddle wheel 需要这些指令集。

**验证方法：**
```bash
python -c "import paddle; print(paddle.device.get_device())"
```

如果提示不支持某些指令，需要重新安装对应版本的 paddle。

### 3. 内存不足

多次并发请求/PDF 转图片占用内存未释放，导致推理时内存不足。

**验证方法：**
在 Windows 任务管理器中看 python 进程内存占用是否持续增长。

---

## 立即可行的缓解措施

### 方案 1：强制限制 PDF 转图片的 DPI

在 `_convert_pdf_to_image` 中，当前 DPI 是 300，可以降低到 200 或 150：

```python
images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
```

### 方案 2：转图片后立即检查尺寸并降采样

在 `_convert_pdf_to_image` 返回前，加一段：

```python
# 如果图片过大，立即降采样
from PIL import Image
img = Image.open(tmp.name)
w, h = img.size
max_dim = 2400  # 限制最大边
if max(w, h) > max_dim:
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    img.save(tmp.name)
    logger.info(f"PDF 转图片后降采样: {(w,h)} -> {(new_w,new_h)}")
```

### 方案 3：给 PaddleOCR 限制线程数

在 `paddle_ocr_service.py` 初始化前加：

```python
import paddle
paddle.set_num_threads(2)  # 限制为 2 线程，降低资源竞争
```

---

## 长期根治方案

### 1. 升级硬件/迁移到 Linux

Windows + CPU 推理本身就不如 Linux 稳定。建议：
- 迁移到 Linux（Ubuntu 20.04/22.04）
- 给机器加内存（至少 8GB）
- 按 `DEPLOY_LINUX.md` 配置生产环境

### 2. 换用更轻量的 OCR 引擎

如果 PaddleOCR 在你这套环境下持续不稳定，可以考虑：
- Tesseract OCR（更轻量，但中文识别率不如 PaddleOCR）
- EasyOCR（基于 PyTorch，可能在某些环境下更稳定）

### 3. 彻底隔离 OCR 服务

把 OCR 拆成独立服务（另一个进程/容器），通过 HTTP 调用：
- 主服务只负责编排
- OCR 服务独立部署、独立重启，崩溃不影响主服务

---

## 当前代码状态总结

✅ **已实现（不依赖 LLM）：**
- PaddleOCRService：三次降级重试（cls=True → cls=False → 降采样+cls=False）
- OCRService：简洁 OCR 失败时自动回退到"预处理+内存识别"
- 详细日志：每次重试/降级都有日志记录

❌ **未实现（按你要求不加）：**
- LLM 多模态兜底（你明确不要）

⚠️ **如果仍然 500：**
- 说明这个 PDF/图片在当前环境下无法被 PaddleOCR 识别
- 需要用上面"立即可行的缓解措施"进一步调整

---

## 下一步行动建议（优先级排序）

1. ✅ **已完成**：在 `_convert_pdf_to_image` 加 DPI=200 + 转图片后尺寸检查（限制最大边 2400px）
2. **验证做**：看日志里图片尺寸，确认是否超大
3. **环境做**：检查 paddle 版本与 CPU 指令集是否匹配
4. **长期做**：迁移到 Linux 或隔离 OCR 服务

---

## 已实施的立即缓解措施（2025-12-26）

### 1. PDF 转图片 DPI 降低：300 → 200

**位置：** `app/services/final_ai_service.py -> _convert_pdf_to_image`

```python
images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
```

**效果：** 减少约 44% 的像素数量，显著降低内存占用与推理引擎负担。

### 2. 转图片后尺寸检查与强制降采样

**位置：** 同上

```python
w, h = img.size
max_dim = 2400  # 限制最大边
if max(w, h) > max_dim:
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    logger.warning("PDF 转图片后尺寸过大，已降采样", ...)
```

**效果：** 确保进入 OCR 的图片最大边不超过 2400px，避免后续 primitive 执行失败。

---

## 验证步骤

下次遇到 PDF 请求时，检查日志：

- 如果看到 `PDF 转图片尺寸正常`：说明原图就不大，问题可能在 CPU/内存
- 如果看到 `PDF 转图片后尺寸过大，已降采样`：说明原图超大，已被限制

如果仍然失败，请查看：
- 降采样后的 `final_size`
- 任务管理器中 python 进程内存占用

