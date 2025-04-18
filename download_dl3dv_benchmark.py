""" This script is used to download the DL3DV benchmark from the huggingface repo.
    The benchmark is composed of 140 different scenes covering different scene complexities (reflection, transparency, indoor/outdoor, etc.) 
    The whole benchmark is very large: 2.1 TB. So we provide this script to download the subset of the dataset based on common needs. 
        - [x] Full benchmark downloading
            Full download can directly be done by git clone (w. lfs installed).
        - [x] scene downloading based on scene hash code  
        Option: 
        - [x] images_8 (480 x 270 resolution) level dataset
"""

import os 
from os.path import join, exists, dirname
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi 
import argparse
import traceback
import pickle
import shutil
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging



# # 设置Hugging Face镜像站点
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 禁用所有进度条
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TQDM"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.hf_api").setLevel(logging.WARNING)

# 禁用tqdm的进度条
tqdm.pandas(disable=True)

api = HfApi()
repo_root = 'DL3DV/DL3DV-10K-Benchmark'
# 全局锁，用于保护共享资源
file_lock = Lock()
hash_lock = Lock()
print_lock = Lock()


def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs)


def hf_download_path(repo_path: str, odir: str, max_try: int = 10, timeout: int = 30):
    """ hf api is not reliable, retry when failed with max tries
    :param repo_path: The path of the repo to download
    :param odir: output path 
    :param max_try: max number of retries
    :param timeout: timeout in seconds
    """	
    rel_path = os.path.relpath(repo_path, repo_root)
    
    # 检查目标文件是否已经存在
    target_file = join(odir, rel_path)
    with file_lock:
        if exists(target_file) and os.path.getsize(target_file) > 0:
            return True
        
        # 确保目标目录存在
        os.makedirs(dirname(target_file), exist_ok=True)

    counter = 0
    while True:
        if counter >= max_try:
            return False

        try:
            # 禁用进度条显示
            api.hf_hub_download(
                repo_id=repo_root, 
                filename=rel_path, 
                repo_type='dataset', 
                local_dir=odir, 
                cache_dir=join(odir, '.cache'),
                local_dir_use_symlinks=False,  # 禁用进度条
                force_download=True  # 强制下载，避免进度条
            )
            return True

        except Exception as e:
            counter += 1
            retry_wait = min(2 ** counter + random.uniform(0, 1), 60)  # 指数退避策略
            time.sleep(retry_wait)


def clean_huggingface_cache(cache_dir: str):
    """ Huggingface cache may take too much space, we clean the cache to save space if necessary
    :param cache_dir: the current cache directory 
    """    
    # Current huggingface hub does not provide good practice to clean the space.  
    # We mannually clean the cache directory if necessary.
    cache_path = join(cache_dir, 'datasets--DL3DV--DL3DV-10K-Benchmark')
    if exists(cache_path):
        try:
            shutil.rmtree(cache_path)
            safe_print(f"缓存目录已清理: {cache_path}")
        except Exception as e:
            safe_print(f"清理缓存失败: {e}")
    else:
        safe_print(f"缓存目录不存在，跳过清理: {cache_path}")


def get_checkpoint_path(output_dir: str):
    """ 获取断点续传的检查点文件路径
    :param output_dir: 输出目录
    :return: 检查点文件路径
    """
    return join(output_dir, '.download_checkpoint.json')


def get_file_checkpoint_path(output_dir: str):
    """ 获取文件级断点续传的检查点文件路径
    :param output_dir: 输出目录
    :return: 文件级检查点文件路径
    """
    return join(output_dir, '.file_download_checkpoint.json')


def load_download_checkpoint(output_dir: str):
    """ 加载已下载的哈希记录
    :param output_dir: 输出目录
    :return: 已下载哈希集合
    """
    checkpoint_path = get_checkpoint_path(output_dir)
    if exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            safe_print(f"加载断点续传数据失败: {e}")
    return set()


def load_file_checkpoint(output_dir: str):
    """ 加载已下载的文件记录
    :param output_dir: 输出目录
    :return: 已下载文件集合
    """
    checkpoint_path = get_file_checkpoint_path(output_dir)
    if exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            safe_print(f"加载文件断点续传数据失败: {e}")
    return set()


def save_download_checkpoint(output_dir: str, downloaded_hashes: set):
    """ 保存已下载的哈希记录
    :param output_dir: 输出目录
    :param downloaded_hashes: 已下载哈希集合
    """
    checkpoint_path = get_checkpoint_path(output_dir)
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(list(downloaded_hashes), f)
    except Exception as e:
        safe_print(f"保存断点续传数据失败: {e}")


def save_file_checkpoint(output_dir: str, downloaded_files: set):
    """ 保存已下载的文件记录
    :param output_dir: 输出目录
    :param downloaded_files: 已下载文件集合
    """
    checkpoint_path = get_file_checkpoint_path(output_dir)
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(list(downloaded_files), f)
    except Exception as e:
        safe_print(f"保存文件断点续传数据失败: {e}")


def download_by_hash(filepath_dict: dict, odir: str, hash: str, only_level8: bool, downloaded_files: set, force: bool = False):
    """ Given a hash, download the relevant data from the huggingface repo 
    :param filepath_dict: the cache dict that stores all the file relative paths 
    :param odir: the download directory 
    :param hash: the hash code for the scene 
    :param only_level8: the images_8 resolution level, if true, only the images_8 resolution level will be downloaded 
    :param downloaded_files: 已下载的文件集合
    :param force: 是否强制重新下载
    """	
    all_files = filepath_dict[hash]
    download_files = [join(repo_root, f) for f in all_files] 

    if only_level8: # only download images_8 level data
        download_files = []
        for f in all_files:
            subdirname = os.path.basename(os.path.dirname(f))

            if 'images' in f and subdirname != 'images_8' or 'input' in f:
                continue 

            download_files.append(join(repo_root, f))

    # 过滤掉已下载的文件
    if not force:
        with file_lock:
            remaining_files = [f for f in download_files if f not in downloaded_files]
            download_files = remaining_files

    if not download_files:
        return True

    for f in download_files:
        if hf_download_path(f, odir):
            with file_lock:
                downloaded_files.add(f)
                # 每下载10个文件保存一次断点记录
                if len(downloaded_files) % 10 == 0:
                    save_file_checkpoint(odir, downloaded_files)
        else:
            with file_lock:
                save_file_checkpoint(odir, downloaded_files)
            return False
            
    with file_lock:
        save_file_checkpoint(odir, downloaded_files)
    return True


def download_scene(args, filepath_dict, hash_value, downloaded_files, downloaded_hashes):
    """下载单个场景的线程函数"""
    try:
        if download_by_hash(filepath_dict, args.odir, hash_value, args.only_level8, downloaded_files, args.force):
            with hash_lock:
                downloaded_hashes.add(hash_value)
                save_download_checkpoint(args.odir, downloaded_hashes)
            return True
        return False
    except Exception as e:
        return False


def download_benchmark(args):
    """ Download the benchmark based on the user inputs.
        1. download the benchmark-meta.csv
        2. based on the args, download the specific subset 
            a. full benchmark 
            b. full benchmark in images_8 resolution level 
            c. full benchmark only with nerfstudio colmaps (w.o. gaussian splatting colmaps) 
            d. specific scene based on the index in [0, 140)
    :param args: argparse args. Used to decide the subset.
    :return: download success or not
    """	
    output_dir = args.odir
    subset_opt = args.subset
    level8_opt = args.only_level8
    hash_name  = args.hash
    is_clean_cache = args.clean_cache
    resume = args.resume
    restart_hash = args.restart_hash
    force = args.force
    num_workers = args.num_workers

    os.makedirs(output_dir, exist_ok=True)
    
    # 加载断点续传数据
    downloaded_hashes = set()
    downloaded_files = set()
    if resume:
        downloaded_hashes = load_download_checkpoint(output_dir)
        downloaded_files = load_file_checkpoint(output_dir)
        print(f"已下载 {len(downloaded_hashes)} 个场景")

    # STEP 1: download the benchmark-meta.csv and .cache/filelist.bin
    meta_repo_path = join(repo_root, 'benchmark-meta.csv')
    cache_file_path = join(repo_root, '.cache/filelist.bin')
    
    meta_file_path = join(output_dir, 'benchmark-meta.csv')
    cache_bin_path = join(output_dir, '.cache/filelist.bin')
    
    # 如果元数据文件不存在，则下载
    if not exists(meta_file_path) or force:
        if hf_download_path(meta_repo_path, output_dir) == False:
            print('ERROR: Download benchmark-meta.csv failed.')
            return False
    
    # 如果缓存文件不存在，则下载
    os.makedirs(join(output_dir, '.cache'), exist_ok=True)
    if not exists(cache_bin_path) or force:
        if hf_download_path(cache_file_path, output_dir) == False:
            print('ERROR: Download .cache/filelist.bin failed.')
            return False

    # STEP 2: download the specific subset
    df = pd.read_csv(join(output_dir, 'benchmark-meta.csv'))
    filepath_dict = pickle.load(open(join(output_dir, '.cache/filelist.bin'), 'rb'))
    hashlist = df['hash'].tolist()
    download_list = hashlist

    # sanity check 
    if subset_opt == 'hash':  
        if hash_name not in hashlist: 
            print(f'ERROR: hash {hash_name} not in the benchmark-meta.csv')
            return False

        # if subset is hash, only download the specific hash
        download_list = [hash_name]
        # 单场景下载模式下，如果指定了force，则清空已下载记录
        if force:
            # 移除该场景已下载的文件记录
            downloaded_files = set([f for f in downloaded_files if hash_name not in f])
            if hash_name in downloaded_hashes:
                downloaded_hashes.remove(hash_name)

    # 是否从指定场景重新开始
    if restart_hash:
        if restart_hash not in hashlist:
            print(f'ERROR: restart hash {restart_hash} not in the benchmark-meta.csv')
            return False
        
        idx = hashlist.index(restart_hash)
        download_list = hashlist[idx:]
        print(f"从场景 {restart_hash} 开始下载，跳过前 {idx} 个场景")

    # 过滤掉已下载的场景
    if resume and downloaded_hashes and not force:
        remaining_hashes = [h for h in download_list if h not in downloaded_hashes]
        download_list = remaining_hashes

    if not download_list:
        print("没有需要下载的场景")
        return True
    
    # 使用线程池下载场景
    print(f"使用 {num_workers} 个线程下载 {len(download_list)} 个场景...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有下载任务
        future_to_hash = {
            executor.submit(download_scene, args, filepath_dict, hash_value, downloaded_files, downloaded_hashes): hash_value 
            for hash_value in download_list
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(download_list), desc="下载进度") as pbar:
            for future in as_completed(future_to_hash):
                hash_value = future_to_hash[future]
                try:
                    success = future.result()
                    if success:
                        pbar.update(1)
                        if is_clean_cache:
                            clean_huggingface_cache(join(output_dir, '.cache'))
                    else:
                        print(f"场景 {hash_value} 下载失败")
                        return False
                except Exception as e:
                    print(f"场景 {hash_value} 下载出错: {e}")
                    return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', type=str, help='output directory', default='DL3DV-10K-Benchmark')
    parser.add_argument('--subset', choices=['full', 'hash'], help='The subset of the benchmark to download', required=True)
    parser.add_argument('--only_level8', action='store_true', help='If set, only the images_8 resolution level will be downloaded to save space')
    parser.add_argument('--clean_cache', action='store_true', help='If set, will clean the huggingface cache to save space')
    parser.add_argument('--hash', type=str, help='If set subset=hash, this is the hash code of the scene to download', default='')
    parser.add_argument('--resume', action='store_true', help='If set, will resume download from last checkpoint')
    parser.add_argument('--force', action='store_true', help='If set, will force re-download all files')
    parser.add_argument('--restart_hash', type=str, help='If set, will restart download from the specified hash', default='')
    parser.add_argument('--num_workers', type=int, help='Number of parallel download threads', default=16)
    params = parser.parse_args()


    if download_benchmark(params):
        print('下载完成。数据路径:', params.odir)
    else:
        print(f'下载到 {params.odir} 失败。详见错误信息。')
