import os
import subprocess
import re
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 配置参数（根据实际修改）
FFMPEG_PATH = r"C:\ffmpeg-2025-02-26-git-99e2af4e78-essentials_build\bin"
FILE_SIZE_THRESHOLD = 1024  # 预分割阈值（单位MB）
SEGMENT_DURATION = 300      # 预分割时长（秒）
MAX_WORKERS = 4             # 最大并行处理数

# 路径配置
FFMPEG_EXE = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
FFPROBE_EXE = os.path.join(FFMPEG_PATH, "ffprobe.exe")

sys.stdout.reconfigure(encoding='utf-8')

def debug_print(message):
    print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {message}", flush=True)

def get_video_duration(input_path):
    """获取视频时长（秒）"""
    try:
        cmd = [
            FFPROBE_EXE, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=True, text=True, encoding='utf-8')
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        debug_print(f"获取时长失败: {str(e)}")
        return 0

def get_optimal_threads():
    try:
        import multiprocessing
        return max(4, multiprocessing.cpu_count() // 2)
    except:
        return 6

def detect_hardware():
    try:
        result = subprocess.run([FFMPEG_EXE, '-hide_banner', '-encoders'], 
                              capture_output=True, text=True, encoding='utf-8')
        for codec in ['h264_nvenc', 'h264_qsv', 'h264_amf']:
            if codec in result.stdout:
                return codec
        return 'libx264'
    except:
        return None

def split_large_file(input_path, temp_dir):
    """智能分割大文件"""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_pattern = os.path.join(temp_dir, f"{base_name}_part%03d.mp4")
    
    cmd = [
        FFMPEG_EXE, '-y',
        '-i', input_path,
        '-c:v', 'copy', '-c:a', 'copy',
        '-f', 'segment',
        '-segment_time', str(SEGMENT_DURATION),
        '-reset_timestamps', '1',
        '-map', '0',
        output_pattern
    ]
    
    try:
        debug_print(f"开始分割文件: {os.path.basename(input_path)}")
        duration = get_video_duration(input_path)
        if duration <= 0:
            raise ValueError("无效的视频时长")
            
        with tqdm(total=100, desc="文件分割", unit='%') as pbar:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                      universal_newlines=True, encoding='utf-8')
            
            time_pattern = re.compile(r"out_time=(\d+:\d+:\d+\.\d+)")
            
            for line in process.stdout:
                if match := time_pattern.search(line):
                    h, m, s = match.group(1).split(':')
                    current_time = int(h)*3600 + int(m)*60 + float(s)
                    progress = current_time / duration * 100
                    pbar.update(min(progress - pbar.n, 100 - pbar.n))
            
            process.wait()
            return sorted([f for f in os.listdir(temp_dir) if f.startswith(base_name)])
    
    except Exception as e:
        debug_print(f"分割失败: {str(e)}")
        return []

def process_segment(segment_path, output_dir):
    """处理单个分段"""
    base_name = os.path.splitext(os.path.basename(segment_path))[0]
    temp_audio = os.path.join(output_dir, f"temp_{base_name}.wav")
    
    # 音频提取命令
    cmd_extract = [
        FFMPEG_EXE, '-y',
        '-i', segment_path,
        '-vn', '-ar', '16000', '-ac', '1',
        '-acodec', 'pcm_s16le',
        '-filter:a', 'loudnorm',
        '-threads', str(get_optimal_threads()),
        temp_audio
    ]
    
    # 分段命令
    cmd_segment = [
        FFMPEG_EXE, '-y',
        '-i', temp_audio,
        '-f', 'segment',
        '-segment_time', '15',
        '-c', 'copy',
        os.path.join(output_dir, f"{base_name}_%03d.wav")
    ]
    
    try:
        # 执行音频提取
        with tqdm(total=100, desc="音频处理", leave=False) as pbar:
            subprocess.run(cmd_extract, check=True,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.STDOUT)
            pbar.update(50)
            
            # 执行分段
            subprocess.run(cmd_segment, check=True,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.STDOUT)
            pbar.update(50)
            
        os.remove(temp_audio)
        return True
    except Exception as e:
        debug_print(f"处理失败: {os.path.basename(segment_path)} - {str(e)}")
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return False

def process_file(input_path, output_dir):
    """智能处理单个文件"""
    file_size = os.path.getsize(input_path) / (1024 ** 2)
    
    if file_size > FILE_SIZE_THRESHOLD:
        temp_dir = os.path.join(output_dir, "temp_segments")
        os.makedirs(temp_dir, exist_ok=True)
        
        segments = split_large_file(input_path, temp_dir)
        if not segments:
            return False
            
        success = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_segment, os.path.join(temp_dir, s), output_dir) 
                      for s in segments]
            for future in futures:
                if future.result():
                    success += 1
        
        shutil.rmtree(temp_dir)
        return success > 0
    else:
        return process_segment(input_path, output_dir)

def main():
    print("视频处理程序 v2.0（大文件优化版）")
    input_dir = input("请输入视频目录（默认./videos）: ") or "./videos"
    output_dir = input("请输入输出目录（默认./output）: ") or "./output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.mp4','.mkv','.avi'))]
    
    with tqdm(total=len(video_files), desc="总体进度") as main_bar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for vid in video_files:
                input_path = os.path.join(input_dir, vid)
                futures.append(executor.submit(process_file, input_path, output_dir))
            
            for future in futures:
                future.result()
                main_bar.update(1)
    
    print("处理完成！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"严重错误: {str(e)}")
