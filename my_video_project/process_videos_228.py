import os
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
import time
import sys
from tqdm import tqdm  # 新增进度条库

# 设置默认编码为 utf-8
sys.stdout.reconfigure(encoding='utf-8')

def get_video_duration(input_file):
    """获取视频时长"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            input_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return float(result.stdout)
    except:
        return 0

def process_audio_for_funasr(input_file, output_dir, segment_time=15):
    """
    专门处理适合FunASR的音频格式
    参数:
        input_file: 输入视频文件路径
        output_dir: 输出目录
        segment_time: 分段时长(秒)，默认15秒
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 文件检查
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(input_file) / (1024*1024)  # 转换为MB
    print(f"文件大小: {file_size:.2f} MB")
    
    # 获取视频时长
    duration = get_video_duration(input_file)
    print(f"视频时长: {duration:.2f} 秒")
    
    # 生成基于输入文件名的临时文件名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    temp_audio = os.path.join(output_dir, f"temp_{base_name}.wav")
    
    # 第一步：提取并标准化音频
    normalize_command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件而不询问
        '-i', input_file,
        '-vn',                # 禁用视频
        '-ar', '16000',      # 16kHz采样率
        '-ac', '1',          # 单通道
        '-acodec', 'pcm_s16le',
        '-filter:a', 'loudnorm',  # 音量标准化
        '-threads', '4',     # 增加线程数以提高性能
        '-progress', 'pipe:1',  # 将进度输出到标准输出
        temp_audio
    ]
    
    # 第二步：分段处理
    segment_command = [
        'ffmpeg',
        '-y',
        '-i', temp_audio,
        '-f', 'segment',
        '-segment_time', str(segment_time),
        '-c', 'copy',
        '-threads', '4',      # 增加线程数以提高性能
        '-progress', 'pipe:1', # 将进度输出到标准输出
        os.path.join(output_dir, f"{base_name}_%03d.wav")
    ]
    
    try:
        print(f"正在处理文件: {input_file}")
        print("步骤1: 音频提取和标准化...")
        print("执行的音频提取命令:", ' '.join(normalize_command))
        
        # 使用tqdm创建进度条
        with tqdm(total=int(duration), desc="音频提取进度") as pbar:
            process = subprocess.Popen(
                normalize_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                encoding='utf-8'
            )
            
            # 正则表达式用于从FFmpeg输出中提取时间信息
            time_pattern = re.compile(r"out_time=(\d+:\d+:\d+\.\d+)")
            
            # 监控实际进度
            current_time = 0
            for line in process.stdout:
                match = time_pattern.search(line)
                if match:
                    time_str = match.group(1)
                    # 转换为秒
                    h, m, s = time_str.split(':')
                    seconds = float(h) * 3600 + float(m) * 60 + float(s)
                    # 更新进度条
                    if seconds > current_time:
                        pbar.update(int(seconds - current_time))
                        current_time = seconds
                    # 添加预计剩余时间描述
                    pbar.set_description(f"音频提取 ({(seconds/duration*100):.1f}%)")
            
            # 等待进程完成
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, normalize_command)
            
            # 确保进度条完成
            pbar.update(pbar.total - pbar.n)
        
        print("\n步骤2: 音频分段...")
        print("执行的分段命令:", ' '.join(segment_command))
        
        # 分段处理进度条 - 这个通常很快，使用简单进度显示即可
        with tqdm(total=100, desc="音频分段进度") as pbar:
            process = subprocess.Popen(
                segment_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                encoding='utf-8'
            )
            
            # 监控分段进度 - 分段速度通常很快
            for i, line in enumerate(process.stdout):
                if i % 10 == 0:  # 每10行更新一次进度条
                    pbar.update(1)
                    if pbar.n >= 100:  # 确保不会超过总进度
                        break
            
            # 等待进程完成
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, segment_command)
            
            # 确保进度条完成
            pbar.update(pbar.total - pbar.n)
        
        # 清理临时文件
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        print(f"文件 {input_file} 处理完成！")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"处理超时: {input_file}")
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return False
    except subprocess.CalledProcessError as e:
        print(f"处理错误 {input_file}: {str(e)}")
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return False
    except Exception as e:
        print(f"未知错误: {str(e)}")
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return False

def process_multiple_files(input_dir, output_dir, max_workers=4):
    """
    处理目录中的多个视频文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有视频文件
    video_extensions = ('.mp4', '.ts', '.mov', '.avi', '.mkv')
    video_files = [f for f in os.listdir(input_dir) 
                  if os.path.isfile(os.path.join(input_dir, f)) 
                  and f.lower().endswith(video_extensions)]

    if not video_files:
        print("没有找到视频文件！")
        return

    print(f"找到 {len(video_files)} 个视频文件")
    
    # 使用线程池处理多个文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_file in video_files:
            input_path = os.path.join(input_dir, video_file)
            futures.append(
                executor.submit(process_audio_for_funasr, input_path, output_dir)
            )
        
        # 等待所有任务完成并显示进度
        total = len(futures)
        with tqdm(total=total, desc="总体进度") as pbar:
            completed = 0
            for future in futures:
                future.result()
                completed += 1
                pbar.update(1)
                pbar.set_description(f"总进度: {completed}/{total} ({(completed/total*100):.1f}%)")

def main():
    """
    主函数，设置输入文件、输出目录和选项。
    """
    # --- 用户配置区域 ---
    input_dir = "D:/my_video_project/input"  # 输入视频所在的目录
    output_dir = "D:/my_video_project/output"  # 输出文件保存的目录
    
    print("="*50)
    print("视频处理工具")
    print("="*50)
    print("请选择处理模式：")
    print("1. 处理单个文件")
    print("2. 批量处理目录下所有视频")
    choice = input("请输入选择（1或2）: ")

    if choice == "1":
        # 单文件处理
        input_filename = input("请输入视频文件名: ")  # 例如 "video.mp4"
        input_file = os.path.join(input_dir, input_filename)
        
        if not os.path.isfile(input_file):
            print(f"错误：找不到输入文件 '{input_file}'")
            return
            
        process_audio_for_funasr(input_file, output_dir)
    
    elif choice == "2":
        # 批量处理
        max_workers = int(input("请输入同时处理的文件数（建议2-4）: "))
        process_multiple_files(input_dir, output_dir, max_workers)
    
    else:
        print("无效的选择！")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n总处理时间: {(end_time-start_time)/60:.1f} 分钟")
