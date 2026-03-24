import traceback
import os
from paddleocr import PaddleOCR

# ==============================================================================
# 这是一个独立的诊断脚本，用于测试 PaddleOCR 在您的环境中的运行情况。
#
# 如何使用:
# 1. 修改下面的 `FILE_TO_TEST` 变量，将其设置为导致错误的那个PDF或图片文件的【绝对路径】。
# 2. 在您的虚拟环境终端中运行此脚本: python tests/test_ocr.py
# 3. 观察终端的输出，并将所有输出信息（无论是成功还是失败）提供出来。
# ==============================================================================

# --- 请修改这里 ---
# 请将此路径替换为导致错误的那个文件的绝对路径
# 例如: 'D:/document_classification_system_flask/tests/samples/failing_document.pdf'
FILE_TO_TEST = "请在这里输入导致错误的文件的绝对路径"


def run_ocr_test():
    """
    执行独立的 PaddleOCR 诊断测试。
    """
    if not os.path.exists(FILE_TO_TEST) or FILE_TO_TEST == "请在这里输入导致错误的文件的绝对路径":
        print(f"!!!!!! 错误 !!!!!!")
        if FILE_TO_TEST.startswith("请在这里"):
            print("请先修改脚本中的 FILE_TO_TEST 变量，使其指向一个真实存在的文件。")
        else:
            print(f"文件未找到: '{FILE_TO_TEST}'")
        return

    try:
        print("--> 步骤 1: 初始化 PaddleOCR 引擎...")
        # 使用 show_log=True 获取更详细的日志
        # 假设使用 CPU，如果您的环境是 GPU，请设置 use_gpu=True
        ocr_engine = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=True, show_log=True)
        print("--> PaddleOCR 引擎初始化成功！")

        print(f"\n--> 步骤 2: 开始处理文件: {FILE_TO_TEST}...")
        # PaddleOCR 直接支持 PDF 和图片路径
        result = ocr_engine.ocr(FILE_TO_TEST, cls=True)
        
        print("\n--> 步骤 3: OCR 处理成功完成！")
        
        if result and result[0]:
            print("\n--- 识别结果预览 ---")
            for idx, line_info in enumerate(result[0]):
                if idx < 100: # 只打印前10行结果
                    text = line_info[1][0]
                    confidence = line_info[1][1]
                    print(f"行 {idx+1}: {text} (置信度: {confidence:.2f})")
            if len(result[0]) > 10:
                print(f"... (还有 {len(result[0]) - 10} 行结果)")
        else:
            print("--- OCR 未识别出任何文本 ---")

    except Exception as e:
        print("\n!!!!!! 在 OCR 处理过程中发生严重错误 !!!!!!")
        # 打印完整的错误堆栈信息，这对于诊断至关重要
        traceback.print_exc()

if __name__ == "__main__":
    run_ocr_test()
