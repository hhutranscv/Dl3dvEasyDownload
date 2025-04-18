""" This script is used to download the DL3DV-10 dataset for all resolution levels from the huggingface repo.
    As the whole dataset is too large for most users, we provide this script so that you can download the dataset efficiently based on your needs.
    We provide several options to download the dataset (image frames with poses):
        - [X] Resolution level: 4K, 2K, 960P, 480P  
        - [X] Subset of the 10K, e.g. 1K(0~1K), 2K(1K~2K), 3K(2K~3K), etc
        - [X] specific hash 
        - [X] file_type: raw video | images+poses | colmap cache 
    
    * 支持多线程并行下载，提高下载速度
    
    使用说明:
    1. 安装依赖: pip install huggingface_hub pandas tqdm
    2. 运行脚本: python download_dl3dv.py --odir 输出目录 --subset 数据子集 --resolution 分辨率 --file_type 文件类型 [--threads 线程数] [--clean_cache]
    3. 示例: python download_dl3dv.py --odir DL3DV-10K --subset 1K --resolution 480P --file_type images+poses --threads 2 --clean_cache
"""

import os 
from os.path import join
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi 
import argparse
import traceback
import shutil
import urllib.request
import zipfile                                 
from huggingface_hub import HfFileSystem
import threading
import concurrent.futures
import time
import queue
from threading import Lock

# 检查huggingface_hub的版本
import pkg_resources
try:
    hf_version = pkg_resources.get_distribution("huggingface_hub").version
    print(f"检测到 huggingface_hub 版本: {hf_version}")
except Exception as e:
    hf_version = "0.0.0"
    print(f"无法检测 huggingface_hub 版本，假设为旧版本: {e}")

# 根据版本设置适当的下载参数 - 不使用max_retries参数，它似乎在多个版本中都不可用
USE_ADVANCED_PARAMS = False  # 禁用高级参数以确保兼容性

# 设置镜像站点（可选，如果访问原站点较慢）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

api = HfApi()
resolution2repo = {
    '480P': 'DL3DV/DL3DV-ALL-480P',
    '960P': 'DL3DV/DL3DV-ALL-960P',
    '2K': 'DL3DV/DL3DV-ALL-2K',
    '4K': 'DL3DV/DL3DV-ALL-4K'
}

def verify_access(repo: str):
    """ This function can be used to verify if the user has access to the repo. 

    :param repo: the repo name  
    :return: True if the user has access, False otherwise
    """    
    fs = HfFileSystem()
    try:
        fs.ls(f'datasets/{repo}')
        return True
    except BaseException as e:
        return False


def hf_download_path(repo: str, rel_path: str, odir: str, max_try: int = 5):
    """ hf api is not reliable, retry when failed with max tries

    :param repo: The huggingface dataset repo 
    :param rel_path: The relative path in the repo
    :param odir: output path 
    :param max_try: As the downloading is not a reliable process, we will retry for max_try times
    """	
    counter = 0
    # 在每个线程中创建本地的临时文件目录
    thread_id = threading.get_ident()
    local_cache_dir = join(odir, f'.cache_tmp_{thread_id}')
    os.makedirs(local_cache_dir, exist_ok=True)
    
    while True:
        if counter >= max_try:
            print(f"错误: 下载 {repo}/{rel_path} 失败，已达到最大重试次数。")
            return False
        try:
            # 使用最基本的参数以确保兼容性
            download_params = {
                "repo_id": repo,
                "filename": rel_path,
                "repo_type": "dataset",
                "local_dir": odir,
                "cache_dir": local_cache_dir,
                "force_download": True,
                "resume_download": False
            }
            
            # 不再使用条件判断添加高级参数
            # 如果需要额外的重试，我们自己在外层循环中处理
            output_file = api.hf_hub_download(**download_params)
            
            # 成功下载后删除临时目录
            try:
                shutil.rmtree(local_cache_dir, ignore_errors=True)
            except:
                pass
            return True

        except KeyboardInterrupt:
            print('键盘中断。退出。')
            exit()
        except BaseException as e:
            print(f"尝试 {counter+1}/{max_try} 下载 {rel_path} 失败: {str(e)[:100]}...")
            # 尝试清理可能的损坏文件
            try:
                output_file_path = os.path.join(odir, rel_path)
                if os.path.exists(output_file_path) and os.path.getsize(output_file_path) == 0:
                    os.remove(output_file_path)
            except:
                pass
            time.sleep(2 + counter * 2)  # 指数退避策略
            counter += 1


def download_from_url(url: str, ofile: str):
    """ Download a file from the url to ofile 

    :param url: The url link 
    :param ofile: The output path 
    :return: True if download success, False otherwise
    """    
    try:
        # Use urllib.request.urlretrieve to download the file from `url` and save it locally at `local_file_path`
        urllib.request.urlretrieve(url, ofile)
        return True
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}") 
        return False


def clean_huggingface_cache(output_dir: str, repo: str):
    """ Huggingface cache may take too much space, we clean the cache to save space if necessary

        Current huggingface hub does not provide good practice to clean the space.  
        We mannually clean the cache directory if necessary. 

    :param output_dir: the current output directory 
    :param output_dir: the huggingface repo 
    """    
    try:
        repo_cache_dir = repo.replace('/', '--')
        # cur_cache_dir = join(output_dir, '.cache', f'datasets--{repo_cache_dir}')
        cur_cache_dir = join(output_dir, '.cache')

        if os.path.exists(cur_cache_dir):
            # 递归删除可能会很危险，尝试使用更安全的方式
            for root, dirs, files in os.walk(cur_cache_dir, topdown=False):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"无法删除文件 {file}: {e}")
                        
                for dir in dirs:
                    try:
                        dir_path = os.path.join(root, dir)
                        if os.path.exists(dir_path):
                            os.rmdir(dir_path)
                    except Exception as e:
                        print(f"无法删除目录 {dir}: {e}")
                        
            # 最后尝试删除顶层目录
            try:
                if os.path.exists(cur_cache_dir):
                    os.rmdir(cur_cache_dir)
            except Exception as e:
                print(f"无法删除缓存目录: {e}")
    except Exception as e:
        print(f"清理缓存时出错: {e}")
    

def get_download_list(subset_opt: str, hash_name: str, reso_opt: str, file_type: str, output_dir: str):
    """ Get the download list based on the subset and hash name

        1. Get the meta file   
        2. Select the subset. Based on reso_opt, get the downloading list prepared. 
        3. Return the download list.

    :param subset_opt: Subset of the 10K, e.g. 1K(0~1K), 2K(1K~2K), 3K(2K~3K), etc
    :param hash_name: If provided a non-empty string, ignore the subset_opt and only download the specific hash 
    :param reso_opt: The resolution to download. 
    :param file_type: The file type to download: video | images+poses | colmap_cache  
    :param output_dir: The output directory. 
    """    
    def to_download_item(hash_name, reso, batch, file_type):
        if file_type == 'images+poses':
            repo = resolution2repo[reso]
            rel_path = f'{batch}/{hash_name}.zip'
        elif file_type == 'video':
            repo = 'DL3DV/DL3DV-ALL-video'
            rel_path = f'{batch}/{hash_name}/video.mp4'
        elif file_type == 'colmap_cache':
            repo = 'DL3DV/DL3DV-ALL-ColmapCache'
            rel_path = f'{batch}/{hash_name}.zip'

        # return f'{repo}/{batch}/{hash_name}'
        return { 'repo': repo, 'rel_path': rel_path }

    ret = []

    meta_link = 'https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv'
    cache_folder = join(output_dir, '.cache') 
    meta_file = join(cache_folder, 'DL3DV-valid.csv')
    os.makedirs(cache_folder, exist_ok=True)
    if not os.path.exists(meta_file):
        assert download_from_url(meta_link, meta_file), 'Download meta file failed.'

    df = pd.read_csv(meta_file)

    # if hash is set, ignore the subset_opt
    if hash_name != '':
        assert hash_name in df['hash'].values, f'Hash {hash_name} not found in the meta file.'

        batch = df[df['hash'] == hash_name]['batch'].values[0]
        link = to_download_item(hash_name, reso_opt, batch, file_type)
        ret = [link]
        return ret

    # if hash not set, we download the whole subset
    subdf = df[df['batch'] == subset_opt]
    for i, r in subdf.iterrows():
        hash_name = r['hash']
        ret.append(to_download_item(hash_name, reso_opt, subset_opt, file_type))

    return ret


def download_worker(item, output_dir, is_clean_cache, progress_queue, lock, cache_lock):
    """工作线程函数，下载单个文件并更新进度"""
    repo = item['repo']
    rel_path = item['rel_path']
    
    output_path = os.path.join(output_dir, rel_path)
    output_dir_path = os.path.dirname(output_path)
    output_path_no_ext = output_path.replace('.zip', '')
    
    # 确保输出目录存在
    try:
        os.makedirs(output_dir_path, exist_ok=True)
    except Exception as e:
        with lock:
            progress_queue.put((False, f"无法创建目录 {output_dir_path}: {e}"))
        return False
    
    # 如果文件已存在，跳过下载
    if os.path.exists(output_path_no_ext) and (os.path.isdir(output_path_no_ext) or os.path.getsize(output_path_no_ext) > 0):
        with lock:
            progress_queue.put((True, f"文件已存在：{rel_path}"))
        return True
    
    # 执行下载
    succ = hf_download_path(repo, rel_path, output_dir)
    
    if succ:
        # 注意: 在多线程环境下不清理缓存以避免竞争条件
        # 缓存清理将在所有下载完成后进行
        
        # 解压文件
        if rel_path.endswith('.zip'):
            zip_path = output_path
            try:
                if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        ofile = join(output_dir, os.path.dirname(rel_path))
                        zip_ref.extractall(ofile)
                    
                    # 删除zip文件之前确认它存在
                    if os.path.exists(zip_path):
                        try:
                            os.remove(zip_path)
                        except:
                            pass  # 忽略删除错误，这不是致命错误
                    
                    with lock:
                        progress_queue.put((True, f"下载和解压成功：{rel_path}"))
                    return True
                else:
                    with lock:
                        progress_queue.put((False, f"下载的ZIP文件不存在或为空：{zip_path}"))
                    return False
            except Exception as e:
                with lock:
                    progress_queue.put((False, f"解压失败 {rel_path}: {e}"))
                return False
        else:
            with lock:
                progress_queue.put((True, f"下载成功：{rel_path}"))
                return True
    else:
        with lock:
            progress_queue.put((False, f"下载失败：{rel_path}"))
        return False


def download_with_threads(download_list, output_dir, is_clean_cache, num_threads=4):
    """使用多线程并行下载文件

    :param download_list: 要下载的文件列表 [{'repo', 'rel_path'}]
    :param output_dir: 输出目录
    :param is_clean_cache: 是否清理缓存
    :param num_threads: 线程数量
    :return: 是否全部成功下载
    """
    total_files = len(download_list)
    completed_files = 0
    successful_files = 0
    progress_queue = queue.Queue()
    # 使用队列来控制下载任务分配
    task_queue = queue.Queue()
    for item in download_list:
        task_queue.put(item)
    
    lock = Lock()  # 用于保护进度更新
    cache_lock = Lock()  # 专门用于保护缓存清理操作
    
    # 创建进度条
    pbar = tqdm(total=total_files, desc="下载进度")
    
    try:
        # 使用有限数量的线程并排队下载，防止同时下载太多文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 创建工作线程函数
            def worker():
                while True:
                    try:
                        # 非阻塞方式获取任务，如果队列为空则退出
                        try:
                            item = task_queue.get(block=False)
                        except queue.Empty:
                            # 无任务可做，线程退出
                            break
                        
                        # 执行下载任务
                        try:
                            download_worker(item, output_dir, False, progress_queue, lock, cache_lock)
                        except Exception as e:
                            with lock:
                                progress_queue.put((False, f"处理任务 {item['rel_path']} 时发生异常: {e}"))
                        finally:
                            # 无论成功与否，都标记任务为完成
                            task_queue.task_done()
                    except Exception as e:
                        print(f"工作线程发生未捕获异常: {e}")
            
            # 提交工作线程
            futures = []
            for _ in range(num_threads):
                futures.append(executor.submit(worker))
            
            # 创建监视线程，更新进度条
            def progress_monitor():
                nonlocal completed_files, successful_files
                while completed_files < total_files:
                    try:
                        success, message = progress_queue.get(timeout=0.1)
                        
                        completed_files += 1
                        if success:
                            successful_files += 1
                        
                        # 更新进度条
                        pbar.update(1)
                        pbar.set_postfix(成功=f"{successful_files}/{completed_files}")
                        
                        # 显示消息
                        if not success:
                            print(f"\n{message}")
                        
                        progress_queue.task_done()
                    except queue.Empty:
                        # 检查是否所有任务都已提交且处理完毕
                        if task_queue.empty() and all(f.done() for f in futures):
                            # 如果队列为空且所有线程都结束了，但进度不足，说明有任务丢失
                            if completed_files < total_files:
                                print(f"\n警告: 可能有任务丢失。已完成 {completed_files}/{total_files}.")
                                # 填充丢失的进度
                                for _ in range(total_files - completed_files):
                                    progress_queue.put((False, "任务丢失"))
                            else:
                                # 正常退出监视线程
                                break
                        time.sleep(0.1)
                        continue
            
            # 启动监视线程
            monitor_thread = threading.Thread(target=progress_monitor)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # 等待所有任务完成，添加异常处理
            try:
                # 等待任务队列清空
                task_queue.join()
                
                # 等待所有工作线程完成
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                print('\n接收到终止信号，正在关闭线程池...')
                # 清空任务队列，以便工作线程可以退出
                while not task_queue.empty():
                    try:
                        task_queue.get(block=False)
                        task_queue.task_done()
                    except:
                        pass
                executor.shutdown(wait=False)
                raise
                
            # 等待监视线程完成
            monitor_thread.join(timeout=1.0)
            
    except KeyboardInterrupt:
        print("\n下载被用户中断。")
    finally:
        pbar.close()
        
    # 在所有下载完成后清理缓存目录
    if is_clean_cache:
        try:
            print("正在清理缓存目录...")
            cache_dirs = [d for d in os.listdir(output_dir) if d.startswith('.cache')]
            for d in cache_dirs:
                cache_path = os.path.join(output_dir, d)
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path, ignore_errors=True)
            print("缓存清理完成。")
        except Exception as e:
            print(f"清理缓存时出错: {e}")
    
    print(f'下载总结: {successful_files}/{total_files} 个文件成功下载')
    return successful_files >= total_files * 0.95  # 允许有5%的失败


def download_dataset(args):
    """ Download the dataset based on the user inputs.

    :param args: argparse args. Used to decide the subset.
    :return: download success or not
    """	
    output_dir = args.odir
    subset_opt = args.subset
    reso_opt   = args.resolution
    hash_name  = args.hash
    file_type  = args.file_type
    is_clean_cache = args.clean_cache
    num_threads = args.threads

    # 为较大数据集自动调整线程数
    if num_threads > 3 and subset_opt in ['7K', '8K', '9K', '10K', '11K'] and not hash_name:
        original_threads = num_threads
        num_threads = min(num_threads, 3)  # 限制最大线程数为3
        print(f"警告: 对于大型数据集 {subset_opt}，已将线程数从 {original_threads} 自动调整为 {num_threads} 以避免竞争问题。")
        print(f"如果您想使用更多线程，请使用 --hash 参数下载单个场景或使用脚本多次下载不同的子集。")

    os.makedirs(output_dir, exist_ok=True)

    print(f"开始准备下载列表...")
    download_list = get_download_list(subset_opt, hash_name, reso_opt, file_type, output_dir)
    print(f"找到 {len(download_list)} 个文件需要下载")
    
    # 验证授权
    if len(download_list) > 0:
        sample_repo = download_list[0]['repo']
        if not verify_access(sample_repo):
            print(f'您尚未获得访问权限。请前往相关的 huggingface 仓库 (https://huggingface.co/datasets/{sample_repo}) 申请访问权限。')
            return False
            
    try:
        return download_with_threads(download_list, output_dir, is_clean_cache, num_threads)
    except KeyboardInterrupt:
        print("\n下载过程被用户中断。")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DL3DV数据集下载工具 - 支持多线程并行下载")
    parser.add_argument('--odir', type=str, help='输出目录', required=True)
    parser.add_argument('--subset', choices=['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], help='要下载的数据子集', required=True)
    parser.add_argument('--resolution', choices=['4K', '2K', '960P', '480P'], help='下载的分辨率', required=True)
    parser.add_argument('--file_type', choices=['images+poses', 'video', 'colmap_cache'], help='下载的文件类型', required=True)
    parser.add_argument('--hash', type=str, help='如果设置，则只下载指定哈希值的场景', default='')
    parser.add_argument('--clean_cache', action='store_true', help='如果设置，会在下载完成后清理缓存以节省空间')
    parser.add_argument('--threads', type=int, default=4, help='并行下载的线程数量 (建议: 小型数据集2-4, 大型数据集1-2)')
    params = parser.parse_args()

    assert params.file_type in ['images+poses', 'video', 'colmap_cache'], '请检查file_type参数输入是否正确。'

    print(f"\n=== DL3DV数据集下载工具 ===")
    print(f"- 子集: {params.subset}")
    print(f"- 分辨率: {params.resolution}")
    print(f"- 文件类型: {params.file_type}")
    print(f"- 线程数: {params.threads}")
    print(f"- 输出目录: {params.odir}")
    if params.hash:
        print(f"- 仅下载场景: {params.hash}")
    print("="*30)

    try:
        if download_dataset(params):
            print('下载完成。文件保存在', params.odir)
        else:
            print(f'下载到 {params.odir} 失败。请查看错误信息。')
    except KeyboardInterrupt:
        print("\n程序被用户中断。") 