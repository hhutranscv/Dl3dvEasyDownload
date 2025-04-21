#!/usr/bin/env python
"""
脚本用于解压指定目录下所有zip文件到它们原来的位置
支持递归搜索子目录、多线程处理、进度显示和错误处理
"""
   python unzip_files.py C:\path\to\your\folder --workers 8

import os
import zipfile
import argparse
import concurrent.futures
import shutil
from tqdm import tqdm
import sys
import traceback

def extract_zip(zip_path, delete_after=True):
    """
    解压单个zip文件到其所在目录
    
    Args:
        zip_path: zip文件的完整路径
        delete_after: 解压后是否删除原zip文件
        
    Returns:
        tuple: (是否成功, 错误信息如果有)
    """
    try:
        # 获取zip文件所在目录
        extract_dir = os.path.dirname(zip_path)
        
        # 输出解压信息
        filename = os.path.basename(zip_path)
        
        # 检查文件是否有效的zip
        if not zipfile.is_zipfile(zip_path):
            return False, f"{filename} 不是有效的zip文件"
            
        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # 如果需要，删除原始zip文件
        if delete_after:
            os.remove(zip_path)
            
        return True, None
    except Exception as e:
        error_msg = f"解压 {zip_path} 时出错: {str(e)}"
        return False, error_msg

def find_zip_files(base_dir, recursive=True):
    """
    在指定目录中查找所有zip文件
    
    Args:
        base_dir: 要搜索的基础目录
        recursive: 是否递归搜索子目录
        
    Returns:
        list: zip文件路径列表
    """
    zip_files = []
    
    if recursive:
        # 递归搜索
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith('.zip'):
                    zip_files.append(os.path.join(root, file))
    else:
        # 只搜索当前目录
        for file in os.listdir(base_dir):
            if file.lower().endswith('.zip'):
                zip_files.append(os.path.join(base_dir, file))
                
    return zip_files

def process_directory(directory, recursive=True, num_workers=4, delete_after=True):
    """
    处理目录中的所有zip文件
    
    Args:
        directory: 要处理的目录
        recursive: 是否递归搜索子目录
        num_workers: 并行工作线程数
        delete_after: 解压后是否删除原zip文件
    """
    print(f"正在搜索 {directory} 中的zip文件...")
    zip_files = find_zip_files(directory, recursive)
    
    if not zip_files:
        print("未找到任何zip文件!")
        return
        
    print(f"找到 {len(zip_files)} 个zip文件.")
    
    # 创建进度条
    pbar = tqdm(total=len(zip_files), unit='file')
    errors = []
    success_count = 0
    
    # 定义回调函数来更新进度
    def update_progress(future):
        nonlocal success_count
        result, error = future.result()
        if result:
            success_count += 1
        else:
            errors.append(error)
        pbar.update(1)
    
    # 使用线程池处理所有zip文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for zip_file in zip_files:
            future = executor.submit(extract_zip, zip_file, delete_after)
            future.add_done_callback(update_progress)
            futures.append(future)
            
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    pbar.close()
    
    # 显示结果摘要
    print(f"\n解压完成: {success_count}/{len(zip_files)} 个文件成功")
    
    if errors:
        print(f"发生了 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"- {error}")
        if len(errors) > 10:
            print(f"... 以及 {len(errors) - 10} 个更多错误")

def main():
    parser = argparse.ArgumentParser(description='解压指定目录下的所有zip文件')
    parser.add_argument('directory', help='要处理的目录路径')
    parser.add_argument('--non-recursive', action='store_true', help='不递归搜索子目录')
    parser.add_argument('--workers', type=int, default=4, help='并行工作线程数 (默认: 4)')
    parser.add_argument('--keep', action='store_true', help='保留原始zip文件')
    
    args = parser.parse_args()
    
    directory = os.path.abspath(args.directory)
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在.")
        return
    
    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是一个目录.")
        return
    
    try:
        process_directory(
            directory, 
            recursive=not args.non_recursive,
            num_workers=args.workers,
            delete_after=not args.keep
        )
    except KeyboardInterrupt:
        print("\n操作被用户中断!")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 