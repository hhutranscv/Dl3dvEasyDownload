#!/usr/bin/env python
"""
下载DL3DV元数据文件并分析各子集的场景数量
"""

import os
import pandas as pd
import urllib.request
import sys

def download_meta_file(output_path):
    """下载元数据文件"""
    meta_link = 'https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv'
    
    print(f"正在从 {meta_link} 下载元数据文件...")
    try:
        urllib.request.urlretrieve(meta_link, output_path)
        print(f"元数据文件已下载到 {output_path}")
        return True
    except Exception as e:
        print(f"下载元数据文件失败: {e}")
        return False

def analyze_meta_file(meta_path):
    """分析元数据文件中各子集的场景数量"""
    try:
        df = pd.read_csv(meta_path)
        print(f"\n元数据文件包含 {len(df)} 个场景")
        
        # 计算每个batch的场景数量
        batch_counts = df['batch'].value_counts().sort_index()
        
        print("\n各子集的场景数量:")
        print("=" * 30)
        for batch, count in batch_counts.items():
            print(f"{batch}: {count} 个场景")
            
        return df
    except Exception as e:
        print(f"分析元数据文件失败: {e}")
        return None

def check_local_scenes(data_dir, batch, df=None):
    """检查本地下载的场景数量"""
    if df is None:
        return
    
    batch_dir = os.path.join(data_dir, batch)
    if not os.path.exists(batch_dir):
        print(f"\n本地目录 {batch_dir} 不存在")
        return
    
    # 获取该batch在元数据中的所有hash
    batch_hashes = set(df[df['batch'] == batch]['hash'].tolist())
    
    # 获取本地目录中的场景文件夹
    local_scenes = []
    try:
        for item in os.listdir(batch_dir):
            item_path = os.path.join(batch_dir, item)
            if os.path.isdir(item_path):
                local_scenes.append(item)
    except Exception as e:
        print(f"读取本地目录失败: {e}")
    
    print(f"\n本地{batch}目录信息:")
    print(f"元数据中的场景数: {len(batch_hashes)}")
    print(f"本地下载的场景数: {len(local_scenes)}")
    
    missing_scenes = batch_hashes - set(local_scenes)
    if missing_scenes:
        print(f"缺少 {len(missing_scenes)} 个场景")
        print(f"前5个缺少的场景: {list(missing_scenes)[:5]}")

def main():
    if len(sys.argv) < 2:
        print("用法: python check_dl3dv_scenes.py <数据集目录> [子集名称]")
        print("例如: python check_dl3dv_scenes.py /mnt/d/dl3dv-all-480p 8K")
        return
    
    data_dir = sys.argv[1]
    batch = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 创建临时目录
    tmp_dir = os.path.join(os.path.expanduser("~"), ".dl3dv_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    meta_path = os.path.join(tmp_dir, "DL3DV-valid.csv")
    
    # 下载元数据文件
    if not os.path.exists(meta_path) or os.path.getsize(meta_path) == 0:
        if not download_meta_file(meta_path):
            return
    else:
        print(f"使用已有的元数据文件: {meta_path}")
    
    # 分析元数据文件
    df = analyze_meta_file(meta_path)
    
    # 检查指定子集的本地场景
    if batch:
        check_local_scenes(data_dir, batch, df)
    
if __name__ == "__main__":
    main() 