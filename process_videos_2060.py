import os
import subprocess
import re
import time
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 配置参数（根据实际修改）
SEGMENT_THRESHOLD = 1024  # 分割阈值(MB)
SEGMENT_DURATION = 300    # 分割时长(秒)
MAX_WORKERS = os.cpu_count() or 4
FFMPEG_PATH = r"C:\Users\Administrator\Documents\coze\ffmpeg-master-latest-win64-gpl-shared\bin"  # 已验证路径

# 系统配置
sys.stdout.reconfigure(encoding='utf-8')
FFMPEG_EXE = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
FFPROBE_EXE = os.path.join(FFMPEG_PATH, "ffprobe.exe")

class VideoProcessor:
    def __init__(self):
        self.progress_bars = {}
    
    def debug_print(self, message):
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {message}", flush=True)

    def safe_path(self, path):
        """处理中文路径问题"""
        return '\\\\?\\' + os.path.abspath(path) if os.name == 'nt' else path

    def get_video_duration(self, input_file):
        """获取视频精确时长"""
        try:
            safe_path = self.safe_path(input_file)
            cmd = [FFPROBE_EXE, '-v', 'error', '-show_entries', 
                  'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', safe_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
            return float(result.stdout.strip())
        except Exception as e:
            self.debug_print(f"时长获取失败: {str(e)} - 文件路径: {input_file}")
            return 0

    def split_large_video(self, input_file, temp_dir):
        """智能分割大视频文件"""
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_pattern = os.path.join(temp_dir, f"{base_name}_part%03d.mp4")
        
        split_cmd = [
            FFMPEG_EXE, '-y', '-i', self.safe_path(input_file),
            '-c', 'copy', '-f', 'segment',
            '-segment_time', str(SEGMENT_DURATION),
            '-reset_timestamps', '1',
            '-map', '0', self.safe_path(output_pattern)
        ]
        
        try:
            self.debug_print(f"开始分割: {os.path.basename(input_file)}")
            duration = self.get_video_duration(input_file)
            if duration < 1:
                self.debug_print("视频时长异常，尝试强制分割")
                duration = SEGMENT_DURATION * 2
            
            with tqdm(total=100, desc="文件分割", leave=False) as pbar:
                process = subprocess.Popen(
                    split_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, encoding='utf-8'
                )
                
                time_pattern = re.compile(r"out_time=(\d+:\d+:\d+\.\d+)")
                for line in process.stdout:
                    if match := time_pattern.search(line):
                        h, m, s = match.group(1).split(':')
                        current = int(h)*3600 + int(m)*60 + float(s)
                        if duration > 0:
                            pbar.update(int((current / duration) * 100) - pbar.n)
                
                process.wait()
                return sorted([f for f in os.listdir(temp_dir) if f.startswith(base_name)])
        except Exception as e:
            self.debug_print(f"分割失败: {str(e)}")
            return []

    def process_audio_segment(self, input_file, output_dir):
        """处理单个音频分段"""
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        temp_audio = os.path.join(output_dir, f"temp_{base_name}.wav")
        final_pattern = os.path.join(output_dir, f"{base_name}_%03d.wav")
        
        # 提取音频命令
        extract_cmd = [
            FFMPEG_EXE, '-y', '-i', self.safe_path(input_file),
            '-vn', '-ar', '16000', '-ac', '1',
            '-acodec', 'pcm_s16le', '-filter:a', 'loudnorm',
            '-threads', '0', '-stats', '-progress', 'pipe:1',
            self.safe_path(temp_audio)
        ]
        
        # 分割音频命令
        split_cmd = [
            FFMPEG_EXE, '-y', '-i', self.safe_path(temp_audio),
            '-f', 'segment', '-segment_time', '15',
            '-c', 'copy', '-reset_timestamps', '1',
            self.safe_path(final_pattern)
        ]
        
        try:
            with tqdm(total=100, desc="音频处理", leave=False) as pbar:
                # 提取音频
                process = subprocess.Popen(
                    extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, encoding='utf-8'
                )
                duration = self.get_video_duration(input_file) or 1
                time_pattern = re.compile(r"out_time=(\d+:\d+:\d+\.\d+)")
                
                for line in process.stdout:
                    if match := time_pattern.search(line):
                        h, m, s = match.group(1).split(':')
                        current = int(h)*3600 + int(m)*60 + float(s)
                        pbar.update(int((current / duration) * 50) - pbar.n)
                
                # 分割音频
                process = subprocess.Popen(
                    split_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    universal_newlines=True, encoding='utf-8'
                )
                for line in process.stdout:
                    if match := time_pattern.search(line):
                        h, m, s = match.group(1).split(':')
                        current = int(h)*3600 + int(m)*60 + float(s)
                        pbar.update(int((current / duration) * 50) - pbar.n)
                
                os.remove(temp_audio)
                return True
        except Exception as e:
            self.debug_print(f"处理失败: {str(e)}")
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return False

    def process_file(self, input_path, output_dir):
        """主处理流程"""
        try:
            file_size = os.path.getsize(input_path) / (1024 ** 2)
            
            if file_size > SEGMENT_THRESHOLD:
                temp_dir = os.path.join(output_dir, "temp_split")
                os.makedirs(temp_dir, exist_ok=True)
                
                segments = self.split_large_video(input_path, temp_dir)
                if not segments:
                    return False
                
                try:
                    for seg in segments:
                        seg_path = os.path.join(temp_dir, seg)
                        if not self.process_audio_segment(seg_path, output_dir):
                            return False
                    return True
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                return self.process_audio_segment(input_path, output_dir)
        except Exception as e:
            self.debug_print(f"文件处理异常: {str(e)}")
            return False

def main():
    processor = VideoProcessor()
    
    print("视频处理工具（大文件优化版）")
    input_dir = input("请输入视频目录（默认./videos）: ") or "./videos"
    output_dir = input("请输入输出目录（默认./output）: ") or "./output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频文件列表（支持中文字符）
    try:
        video_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.mp4','.mkv','.avi','.mov'))
        ]
    except UnicodeDecodeError:
        video_files = [
            f.encode('utf-8').decode('gbk') for f in os.listdir(input_dir)
            if f.lower().endswith(('.mp4','.mkv','.avi','.mov'))
        ]
    
    with tqdm(total=len(video_files), desc="总体进度") as main_bar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for video in video_files:
                input_path = os.path.join(input_dir, video)
                futures.append(executor.submit(processor.process_file, input_path, output_dir))
            
            for future in futures:
                future.result()
                main_bar.update(1)
    
    print("处理完成！输出文件位于：", os.path.abspath(output_dir))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        print(f"致命错误: {str(e)}")
