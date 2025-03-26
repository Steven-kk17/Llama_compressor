#!/usr/bin/env python3
# filepath: /home/steven/code/llama_compression/run_lora_bpp_tests.py
import subprocess
import re
import os
import csv
import sys
from datetime import datetime
import argparse

# 定义要测试的数据集
# DATASETS = ['kodak', 'div2k', 'clic_mobile', 'clic_professional']
DATASETS = ['clic_mobile', 'clic_professional']

def run_lora_tests(args):
    # 结果保存路径
    results_file = args.output
    results_csv = results_file.replace(".txt", ".csv")
    
    # 写死参数: 始终跳过算术编码和不保存图像
    skip_ac = True
    no_save_images = True
    
    print(f"开始LoRA测试")
    print(f"使用固定参数: --skip_ac --no_save_images {'--keep_original_size' if args.keep_original_size else ''}")
    print(f"结果将保存到: {results_file}")
    
    # 清空或创建结果文件
    with open(results_file, "w") as f:
        f.write(f"# LoRA BPP Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 参数: --skip_ac --no_save_images {'--keep_original_size' if args.keep_original_size else ''}\n\n")
        f.write("Dataset            | Theoretical BPP | Images  \n")
        f.write("-------------------|----------------|--------\n")
    
    # 准备CSV结果文件
    with open(results_csv, "w", newline='') as csvfile:
        fieldnames = ["Dataset", "Theoretical BPP", "Images"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # 为每个数据集运行测试
    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"测试数据集: {dataset}")
        print(f"{'='*60}\n")
        
        # 准备命令行参数
        cmd = [
            "python", "lora_test.py",
            "--dataset_name", dataset,
            "--batch_size", str(args.batch_size),
            "--skip_ac",           # 写死: 始终跳过算术编码
            "--no_save_images"     # 写死: 始终不保存图像
        ]
        
        # 添加可选参数
        if args.keep_original_size:
            cmd.append("--keep_original_size")
            
        if args.single_gpu:
            cmd.append("--single_gpu")
            
        # 打印运行的命令
        print(f"执行命令: {' '.join(cmd)}\n")
        
        # 使用Popen实时捕获输出
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 存储所有输出用于之后提取数据
        full_output = ""
        
        # 实时显示输出
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # 实时打印每一行
            sys.stdout.flush()   # 确保立即显示
            full_output += line
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"\n警告: 命令执行返回代码 {return_code}")
        
        # 提取重要信息
        theor_bpp_pattern = r"Overall Theoretical Dataset BPP: (\d+\.\d+)"
        image_count_pattern = r"Found (\d+) images"
        
        # 优先从输出中提取值
        theor_bpp_match = re.search(theor_bpp_pattern, full_output)
        image_count_match = re.search(image_count_pattern, full_output)
        
        # 如果输出中无法找到，尝试从CSV中提取
        if not theor_bpp_match or not image_count_match:
            csv_file = "./lora_results/lora_metrics.csv"
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    csv_content = f.read()
                    # 提取Model BPP列的值
                    model_bpp_pattern = r'\d+,[^,]*,([0-9.]+)'
                    model_bpps = re.findall(model_bpp_pattern, csv_content)
                    
                    if model_bpps:
                        # 计算平均值
                        theor_bpp = sum(float(bpp) for bpp in model_bpps) / len(model_bpps)
                        image_count = len(model_bpps)
                        print(f"从CSV提取了BPP值: {theor_bpp:.4f}")
                    else:
                        theor_bpp = 0.0
                        image_count = 0
                        print("CSV文件中找不到BPP值")
            else:
                theor_bpp = 0.0
                image_count = 0
                print("找不到测量结果")
        else:
            # 使用输出中的值
            theor_bpp = float(theor_bpp_match.group(1))
            image_count = int(image_count_match.group(1))
        
        # 保存结果到TXT
        with open(results_file, "a") as f:
            f.write(f"{dataset.ljust(19)} | {theor_bpp:.4f}         | {image_count}\n")
        
        # 保存结果到CSV
        with open(results_csv, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([dataset, f"{theor_bpp:.4f}", image_count])
        
        print(f"\n数据集 {dataset} 测试完成")
        print(f"理论 BPP: {theor_bpp:.4f}")
        print(f"图像数: {image_count}")

    print(f"\n所有测试完成！")
    print(f"结果已保存到: {results_file}")
    print(f"CSV结果已保存到: {results_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行LoRA图像压缩测试')
    parser.add_argument('--keep-original-size', action='store_true',
                        help='保持原始图像尺寸')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--single-gpu', action='store_true',
                        help='仅使用单个GPU进行推理')
    parser.add_argument('--output', type=str, default="lora_bpp_results.txt",
                        help='结果输出文件路径')
    
    args = parser.parse_args()
    run_lora_tests(args)
