#!/usr/bin/env python3
import subprocess
import re
import os
import csv
import sys
from datetime import datetime
import argparse

# 定义要测试的数据集
DATASETS = ['kodak', 'div2k', 'clic_mobile', 'clic_professional']

def run_tests(args):
    # 结果保存路径 
    results_file = args.output
    results_csv = results_file.replace(".txt", ".csv")
    
    # 写死参数: 始终跳过算术编码和不保存图像
    skip_ac = True
    no_save_images = True
    
    print(f"开始测试: GPT2 LoRA 模型测试")
    print(f"使用固定参数: --skip_ac --no_save_images")
    print(f"批处理大小: {args.batch_size}")
    print(f"结果将保存到: {results_file}")
    
    # 清空或创建结果文件
    with open(results_file, "w") as f:
        f.write(f"# GPT2 LoRA Model BPP Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 参数: --skip_ac --no_save_images\n")
        f.write(f"# 批处理大小: {args.batch_size}\n\n")
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
        
        # 准备命令
        cmd = [
            "python", "dl_test.py",
            "--dataset_name", dataset,
            "--batch_size", str(args.batch_size),
            "--skip_ac",            # 写死: 始终跳过算术编码
            "--no_save_images"      # 写死: 始终不保存图像
        ]
        
        # 如果指定了单GPU模式，添加参数
        if args.single_gpu:
            cmd.append("--single_gpu")
            
        # 如果指定了模型路径，添加参数
        if args.model:
            cmd.extend(["--model", args.model])
            
        # 如果指定了LoRA路径，添加参数
        if args.lora_path:
            cmd.extend(["--lora_path", args.lora_path])
            
        # 如果指定了GPT2模型目录，添加参数
        if args.gpt2_model_dir:
            cmd.extend(["--gpt2_model_dir", args.gpt2_model_dir])
        
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
        
        # 尝试从输出中提取重要信息
        theor_bpp_pattern = r"Overall Theoretical Dataset BPP: (\d+\.\d+)"
        avg_bpp_pattern = r"Average Model-Estimated BPP: (\d+\.\d+)"
        avg_bpp_pattern2 = r"平均 BPP: (\d+\.\d+)"
        image_count_pattern = r"Found (\d+) images"
        image_count_pattern2 = r"找到 (\d+) 个图像"
        
        # 首先尝试找Overall值，如果找不到，再尝试找Average值
        theor_bpp_match = re.search(theor_bpp_pattern, full_output)
        if not theor_bpp_match:
            theor_bpp_match = re.search(avg_bpp_pattern, full_output)
        if not theor_bpp_match:
            theor_bpp_match = re.search(avg_bpp_pattern2, full_output)
        
        image_count_match = re.search(image_count_pattern, full_output)
        if not image_count_match:
            image_count_match = re.search(image_count_pattern2, full_output)
        
        # 确保获取到了值
        theor_bpp = float(theor_bpp_match.group(1)) if theor_bpp_match else 0.0
        image_count = int(image_count_match.group(1)) if image_count_match else 0
        
        # 如果我们仍然没有找到BPP，尝试在CSV文件中查找
        if theor_bpp == 0.0:
            csv_files = [
                f"./gpt2_lora_results/lora_{dataset}_metrics.csv"
            ]
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    print(f"从CSV文件获取BPP: {csv_file}")
                    with open(csv_file, 'r') as f:
                        csv_content = f.read()
                        # 提取Model BPP列的值，假设是CSV的第3列
                        model_bpp_pattern = r'\d+,[^,]*,([0-9.]+)'
                        model_bpps = re.findall(model_bpp_pattern, csv_content)
                        
                        if model_bpps:
                            # 计算平均值
                            theor_bpp = sum(float(bpp) for bpp in model_bpps) / len(model_bpps)
                            if image_count == 0:
                                image_count = len(model_bpps)
                            print(f"从CSV计算得到平均BPP: {theor_bpp:.4f}")
                            break
        
        # 保存结果到TXT
        with open(results_file, "a") as f:
            f.write(f"{dataset.ljust(19)} | {theor_bpp:.4f}         | {image_count}\n")
        
        # 保存结果到CSV
        with open(results_csv, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([dataset, f"{theor_bpp:.4f}", image_count])
        
        print(f"\n数据集 {dataset} 测试完成")
        print(f"理论 BPP: {theor_bpp:.4f}")
        print(f"图像数量: {image_count}")

    print(f"\n所有测试完成！")
    print(f"结果已保存到: {results_file}")
    print(f"CSV结果已保存到: {results_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行GPT2 LoRA模型图像压缩测试')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--single-gpu', action='store_true',
                        help='仅使用单个GPU进行测试(避免内存问题)')
    parser.add_argument('--output', type=str, default="dl_bpp_results.txt",
                        help='结果输出文件路径')
    parser.add_argument('--model', type=str, default=None,
                        help='蒸馏基础模型路径，不指定则使用默认路径')
    parser.add_argument('--lora-path', type=str, default=None,
                        help='LoRA权重路径，不指定则使用默认路径')
    parser.add_argument('--gpt2-model-dir', type=str, default=None,
                        help='GPT2模型目录路径，不指定则使用默认路径')
    
    args = parser.parse_args()
    run_tests(args)
