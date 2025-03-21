#!/usr/bin/env python3
import subprocess
import re
import os
from datetime import datetime

# 定义要测试的数据集
datasets = ['kodak', 'div2k', 'clic_mobile', 'clic_professional']

# 结果保存路径
results_file = "distill_bpp_results.txt"

# 清空或创建结果文件
with open(results_file, "w") as f:
    f.write(f"# Distilled Model BPP Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("# 参数: --skip_ac --no_save_images, 不保留原始尺寸\n\n")
    f.write("Dataset            | Avg BPP   | Images  \n")
    f.write("-------------------|-----------|--------\n")

# 为每个数据集运行测试
for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"测试数据集: {dataset}")
    print(f"{'='*60}\n")
    
    # 运行命令
    cmd = [
        "python", "distillation_test.py",
        "--dataset_name", dataset,
        "--skip_ac",
        "--no_save_images",
        "--batch_size", "4"
        "--keep_original_size"
    ]
    
    # 执行命令并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 提取所有BPP值和最终平均BPP
    bpp_pattern = r"Average BPP: (\d+\.\d+)"
    image_count_pattern = r"Found (\d+) images"
    
    bpp_values = re.findall(bpp_pattern, result.stdout)
    image_count_match = re.search(image_count_pattern, result.stdout)
    
    # 查找CSV文件中的结果
    csv_file = f"./distilled_results/distilled_metrics.csv"
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            csv_content = f.read()
            # 提取Model BPP列的值
            model_bpp_pattern = r'\d+,"[^"]*",(\d+\.\d+)'
            model_bpps = re.findall(model_bpp_pattern, csv_content)
            
            if model_bpps:
                # 计算平均值
                avg_model_bpp = sum(float(bpp) for bpp in model_bpps) / len(model_bpps)
                image_count = len(model_bpps)
            else:
                # 如果找不到CSV中的值，使用stdout中的值
                avg_model_bpp = sum(float(bpp) for bpp in bpp_values) / len(bpp_values) if bpp_values else 0.0
                image_count = int(image_count_match.group(1)) if image_count_match else 0
    else:
        # 如果找不到CSV文件，使用stdout中的值
        avg_model_bpp = sum(float(bpp) for bpp in bpp_values) / len(bpp_values) if bpp_values else 0.0
        image_count = int(image_count_match.group(1)) if image_count_match else 0
    
    # 保存结果
    with open(results_file, "a") as f:
        f.write(f"{dataset.ljust(19)} | {avg_model_bpp:.4f}    | {image_count}\n")
    
    print(f"\n数据集 {dataset} 测试完成，平均 BPP: {avg_model_bpp:.4f}\n")

print(f"\n所有测试完成！结果已保存到 {results_file}")
