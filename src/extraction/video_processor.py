import os
import time
import json
import shutil
import logging
import requests
import subprocess
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from scipy.io import wavfile

# 设置日志
def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

class AudioProcessor:
    """音频处理类，提供音频分析和处理功能"""
    
    def __init__(self):
        self.logger = logging.getLogger("audio_processor")
    
    def segment_by_silence(self, audio_path, min_length=5, max_length=20):
        """使用静音检测将音频分割成片段"""
        try:
            sound = AudioSegment.from_file(audio_path)
            
            # 使用静音检测分割音频
            chunks = split_on_silence(
                sound,
                min_silence_len=700,  # 静音超过700ms视为分割点
                silence_thresh=sound.dBFS - 14,  # 音量阈值
                keep_silence=300  # 保留300ms静音作为自然过渡
            )
            
            temp_dir = os.path.join(os.path.dirname(audio_path), "chunks")
            os.makedirs(temp_dir, exist_ok=True)
            
            output_files = []
            for i, chunk in enumerate(chunks):
                # 跳过过短的片段
                if len(chunk) < min_length * 1000:
                    continue
                    
                # 分割过长的片段
                if len(chunk) > max_length * 1000:
                    subchunks = self._split_long_chunk(chunk, max_length)
                    for j, subchunk in enumerate(subchunks):
                        chunk_path = os.path.join(temp_dir, f"chunk_{i}_{j}.wav")
                        subchunk.export(chunk_path, format="wav")
                        output_files.append(chunk_path)
                else:
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                    chunk.export(chunk_path, format="wav")
                    output_files.append(chunk_path)
            
            return output_files
        
        except Exception as e:
            self.logger.error(f"分割音频失败: {str(e)}")
            return []
    
    def _split_long_chunk(self, chunk, max_length):
        """将过长的音频片段分割成较小的子片段"""
        max_ms = max_length * 1000
        subchunks = []
        
        # 按最大长度分割
        for i in range(0, len(chunk), max_ms):
            subchunks.append(chunk[i:i + max_ms])
            
        return subchunks
    
    def normalize_audio(self, audio_path, output_path=None):
        """规范化音频音量和格式"""
        if output_path is None:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_normalized{ext}"
        
        try:
            # 加载音频
            audio = AudioSegment.from_file(audio_path)
            
            # 规范化音量
            normalized = self._match_target_amplitude(audio, -18.0)
            
            # 导出规范化后的音频
            normalized.export(output_path, format="wav")
            return output_path
            
        except Exception as e:
            self.logger.error(f"规范化音频失败: {str(e)}")
            return None
    
    def _match_target_amplitude(self, sound, target_dBFS):
        """规范化音频的振幅"""
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)
    
    def validate_audio_file(self, audio_path):
        """验证音频文件是否有效"""
        try:
            audio = AudioSegment.from_file(audio_path)
            if len(audio) < 100:  # 音频太短
                return False
            return True
        except Exception as e:
            self.logger.error(f"音频文件无效: {str(e)}")
            return False

class VideoProcessor:
    def __init__(self, 
                 ffmpeg_path=None,
                 max_workers=4,
                 file_size_threshold=1024,
                 segment_duration=300):
        
        # 默认配置
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"  # 修改为直接使用系统路径中的ffmpeg
        self.file_size_threshold = file_size_threshold  # 单位MB
        self.segment_duration = segment_duration        # 单位秒
        self.max_workers = max_workers
        
        # FFMPEG可执行文件路径
        if os.name == 'nt':  # Windows系统
            self.ffmpeg_exe = os.path.join(self.ffmpeg_path, "ffmpeg.exe") if os.path.isdir(self.ffmpeg_path) else "ffmpeg"
            self.ffprobe_exe = os.path.join(self.ffmpeg_path, "ffprobe.exe") if os.path.isdir(self.ffmpeg_path) else "ffprobe"
        else:  # Linux/Mac系统
            self.ffmpeg_exe = os.path.join(self.ffmpeg_path, "ffmpeg") if os.path.isdir(self.ffmpeg_path) else "ffmpeg"
            self.ffprobe_exe = os.path.join(self.ffmpeg_path, "ffprobe") if os.path.isdir(self.ffmpeg_path) else "ffprobe"
        
        # 配置日志
        self.logger = logging.getLogger("video_processor")
        
    def process_video(self, input_path, output_dir):
        """
        处理视频文件，提取并分割音频
        返回生成的音频片段列表
        """
        self.logger.info(f"开始处理视频: {os.path.basename(input_path)}")
        
        file_size = os.path.getsize(input_path) / (1024 ** 2)
        processed_segments = []
        
        try:
            # 添加进度指示
            start_time = time.time()
            
            # 尝试获取视频时长，如果失败则不显示时长信息
            try:
                duration = self.get_video_duration(input_path)
                print(f"\U0001F3AC 视频信息: {os.path.basename(input_path)} | {file_size:.1f}MB | {duration/60:.1f}分钟")
            except Exception as e:
                self.logger.warning(f"获取视频时长失败: {str(e)}")
                print(f"\U0001F3AC 视频信息: {os.path.basename(input_path)} | {file_size:.1f}MB")
            
            # 如果文件过大，先进行预分割
            if file_size > self.file_size_threshold:
                self.logger.info(f"文件大小 {file_size:.2f}MB 超过阈值，执行预分割")
                print(f"\U0001F4E6 文件过大，正在预分割视频...")
                
                # 创建临时目录
                temp_video_dir = os.path.join(output_dir, "temp_videos")
                os.makedirs(temp_video_dir, exist_ok=True)
                os.chmod(temp_video_dir, 0o777)  # 设置权限
                
                # 预分割大视频
                video_segments = self._split_large_file(input_path, temp_video_dir)
                print(f"✓ 视频分割完成，共 {len(video_segments)} 个片段")
                
                # 处理每个视频片段
                with tqdm(total=len(video_segments), desc="视频片段处理", unit="段") as pbar:
                    for segment in video_segments:
                        segment_path = os.path.join(temp_video_dir, segment)
                        pbar.set_description(f"提取音频: {segment}")
                        audio_path = self._extract_audio(segment_path, output_dir)
                        if audio_path:
                            audio_segments = self._segment_audio(audio_path, output_dir)
                            processed_segments.extend(audio_segments)
                        pbar.update(1)
                    
                # 清理临时视频目录
                shutil.rmtree(temp_video_dir, ignore_errors=True, onerror=self._handle_remove_error)
                
            else:
                # 直接处理小视频文件
                print(f"\U0001F3B5 正在提取音频...")
                audio_path = self._extract_audio(input_path, output_dir)
                if audio_path:
                    print(f"✂️ 正在分割音频为短片段...")
                    processed_segments = self._segment_audio(audio_path, output_dir)
                
            # 处理完成后显示耗时
            elapsed = time.time() - start_time
            print(f"✅ 视频处理完成，耗时: {elapsed:.2f}秒，提取了 {len(processed_segments)} 个音频片段")
            
            return processed_segments
            
        except Exception as e:
            self.logger.error(f"处理视频失败: {str(e)}")
            print(f"❌ 处理视频失败: {str(e)}")
            return []
    
    def _handle_remove_error(self, func, path, exc_info):
        """处理文件删除权限错误"""
        os.chmod(path, 0o777)
        func(path)
            
    def _extract_audio(self, video_path, output_dir):
        """提取音频"""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.wav")
        
        cmd = [
            self.ffmpeg_exe, "-y",
            "-i", video_path,
            "-vn", "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le",
            "-filter:a", "loudnorm",
            output_path
        ]
        
        try:
            self.logger.info(f"提取音频: {os.path.basename(video_path)}")
            process = subprocess.run(cmd, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"提取音频失败: {str(e)}")
            return None
            
    def _segment_audio(self, audio_path, output_dir):
        """将音频分割成片段"""
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        segments_dir = os.path.join(output_dir, f"segments_{base_name}")
        os.makedirs(segments_dir, exist_ok=True)
        
        segment_pattern = os.path.join(segments_dir, f"{base_name}_%03d.wav")
        
        cmd = [
            self.ffmpeg_exe, "-y",
            "-i", audio_path,
            "-f", "segment",
            "-segment_time", "15",  # 15秒一段
            "-c", "copy",
            segment_pattern
        ]
        
        try:
            self.logger.info(f"分割音频: {os.path.basename(audio_path)}")
            subprocess.run(cmd, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            
            # 获取所有生成的片段
            segments = [os.path.join(segments_dir, f) 
                      for f in os.listdir(segments_dir) 
                      if f.endswith(".wav")]
            
            # 删除原始音频文件
            os.remove(audio_path)
            
            return segments
        except subprocess.CalledProcessError as e:
            self.logger.error(f"分割音频失败: {str(e)}")
            return []
    
    def _split_large_file(self, input_path, temp_dir):
        """智能分割大文件"""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_pattern = os.path.join(temp_dir, f"{base_name}_part%03d.mp4")
        
        cmd = [
            self.ffmpeg_exe, '-y',
            '-i', input_path,
            '-c:v', 'copy', '-c:a', 'copy',
            '-f', 'segment',
            '-segment_time', str(self.segment_duration),
            '-reset_timestamps', '1',
            '-map', '0',
            output_pattern
        ]
        
        try:
            self.logger.info(f"开始分割大文件: {os.path.basename(input_path)}")
            subprocess.run(cmd, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            
            # 返回生成的视频片段文件名
            segments = [f for f in os.listdir(temp_dir) 
                      if f.startswith(base_name) and f.endswith('.mp4')]
            return sorted(segments)
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"分割大文件失败: {str(e)}")
            return []

    def get_video_duration(self, input_path):
        """获取视频时长（秒）"""
        try:
            cmd = [
                self.ffprobe_exe, '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   check=True, text=True, encoding='utf-8')
            return float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            self.logger.error(f"获取视频时长失败: {str(e)}")
            # 不返回0，而是抛出异常
            raise ValueError(f"获取视频时长失败: {str(e)}")

class VoiceFilterProcessor:
    """
    声纹过滤处理器，简化版不使用tensorflow-io
    """
    def __init__(self, voice_prints_dir=None):
        self.logger = logging.getLogger("voice_filter")
        
        # 检查GPU可用性
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            self.logger.info(f'使用设备: GPU ({len(self.gpus)} 可用)')
            print(f'使用设备: GPU ({len(self.gpus)} 可用)')
            # 防止内存溢出
            for gpu in self.gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            self.logger.info('使用设备: CPU')
            print('使用设备: CPU')
        
        # 创建简单的声纹特征提取模型
        self.model = self._create_feature_extractor()
        self.logger.info('声纹特征提取模型创建成功')
        
        # 加载主播声纹
        self.voice_prints = {}
        if voice_prints_dir and os.path.exists(voice_prints_dir):
            self.load_voice_prints(voice_prints_dir)
    
    def _create_feature_extractor(self):
        """创建简单的声纹特征提取模型"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16000,)),
            tf.keras.layers.Reshape((-1, 1)),
            tf.keras.layers.Conv1D(64, 400, strides=160, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 400, strides=160, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128)
        ])
        return model
    
    def load_voice_prints(self, voice_prints_dir, pattern=".npy"):
        """加载声纹文件"""
        try:
            streamer_folders = [f for f in os.listdir(voice_prints_dir) 
                              if os.path.isdir(os.path.join(voice_prints_dir, f))]
            
            for streamer in streamer_folders:
                streamer_dir = os.path.join(voice_prints_dir, streamer)
                voice_print_files = [f for f in os.listdir(streamer_dir) if f.endswith(pattern)]
                
                if voice_print_files:
                    latest_file = max(voice_print_files, 
                                     key=lambda f: os.path.getmtime(os.path.join(streamer_dir, f)))
                    voice_print_path = os.path.join(streamer_dir, latest_file)
                    
                    voice_print = np.load(voice_print_path)
                    self.voice_prints[streamer] = voice_print
                    self.logger.info(f'已加载主播 {streamer} 的声纹: {latest_file}')
            
            self.logger.info(f'共加载了 {len(self.voice_prints)} 个主播声纹')
            print(f'已加载 {len(self.voice_prints)} 个主播声纹')
            
        except Exception as e:
            self.logger.error(f'加载声纹文件失败: {e}')
    
    def extract_voice_print(self, audio_path):
        """从音频提取声纹特征"""
        try:
            # 使用librosa加载音频
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # 裁剪或填充到固定长度
            if len(audio) > 16000:
                audio = audio[:16000]
            elif len(audio) < 16000:
                # 填充静音以达到16000采样点
                audio = np.pad(audio, (0, 16000 - len(audio)))
            
            # 提取特征
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            audio_tensor = tf.expand_dims(audio_tensor, axis=0)  # 添加批次维度
            
            embeddings = self.model(audio_tensor)
            return embeddings.numpy()[0]  # 返回numpy数组
        except Exception as e:
            self.logger.error(f'提取声纹特征失败: {e}')
            return None
    
    def compare_voice_prints(self, voice_print1, voice_print2):
        """比较两个声纹的相似度"""
        try:
            # 转换为张量并归一化
            vp1 = tf.nn.l2_normalize(tf.convert_to_tensor(voice_print1, dtype=tf.float32))
            vp2 = tf.nn.l2_normalize(tf.convert_to_tensor(voice_print2, dtype=tf.float32))
            
            # 计算余弦相似度
            similarity = tf.reduce_sum(tf.multiply(vp1, vp2))
            return similarity.numpy()
        except Exception as e:
            self.logger.error(f'声纹比较失败: {e}')
            return 0.0
    
    def filter_audio(self, audio_path, streamer_id, threshold=0.75):
        """比对音频与主播声纹相似度"""
        if streamer_id not in self.voice_prints:
            self.logger.warning(f'未找到主播 {streamer_id} 的声纹')
            return False, 0.0
            
        audio_voice_print = self.extract_voice_print(audio_path)
        if audio_voice_print is None:
            return False, 0.0
            
        similarity = self.compare_voice_prints(audio_voice_print, self.voice_prints[streamer_id])
        is_target_speaker = similarity >= threshold
        return is_target_speaker, similarity

class VoiceDataExtractor:
    def __init__(self, 
                 config_path="config/system_config.json",
                 whisper_api_url=None,
                 min_duration_minutes=10,
                 max_duration_minutes=15,
                 voice_prints_dir=None):  # 添加声纹目录参数
        
        # 初始化配置
        self.whisper_api_url = whisper_api_url or "http://localhost:9000/asr"
        self.min_duration_seconds = min_duration_minutes * 60
        self.max_duration_seconds = max_duration_minutes * 60
        
        # 尝试加载系统配置
        system_config = {}
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    system_config = json.load(f)
                    # 更新配置
                    self.whisper_api_url = whisper_api_url or system_config.get("whisper_api_url", self.whisper_api_url)
        except Exception as e:
            print(f"⚠️ 加载配置文件失败: {str(e)}，使用默认配置")
        
        # 检查GPU可用性
        try:
            import tensorflow as tf
            self.gpus = tf.config.list_physical_devices('GPU')
            if self.gpus:
                print(f'\n\U0001F680 检测到GPU加速: {len(self.gpus)}个设备可用')
                # 防止内存溢出
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                print('\n⚠️ 未检测到GPU，将使用CPU处理')
        except:
            self.gpus = []
            print('\n⚠️ TensorFlow初始化失败，将使用CPU处理')
        
        # 设置多线程参数
        self.max_workers = system_config.get("max_workers", 8)  # 增加默认线程数
        print(f'\U0001F9F5 多线程处理: {self.max_workers}个工作线程\n')
        
        # 初始化处理器
        self.video_processor = VideoProcessor(
            ffmpeg_path=system_config.get("ffmpeg_path"),
            max_workers=self.max_workers,
            file_size_threshold=system_config.get("file_size_threshold", 1024)
        )
        
        self.audio_processor = AudioProcessor()
        
        # 设置日志
        os.makedirs("logs/extraction_logs", exist_ok=True)
        self.logger = setup_logger("extractor", "logs/extraction_logs/extractor.log")
        
        # 创建声纹过滤器（如果提供了声纹目录）
        self.voice_filter = None
        if voice_prints_dir and os.path.exists(voice_prints_dir):
            self.voice_filter = VoiceFilterProcessor(voice_prints_dir)
            self.logger.info(f"声纹过滤器已启用，声纹目录: {voice_prints_dir}")
        else:
            self.logger.info("声纹过滤器未启用")
            
        # 添加API可用性检查
        self.check_api_availability()
        
    def check_api_availability(self):
        """检查Whisper API服务是否可用"""
        try:
            # 尝试获取API根路径
            api_base = self.whisper_api_url.split('/asr')[0]
            health_url = f"{api_base}/health"
            
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                self.logger.info("Whisper API服务正常")
                print("✅ Whisper API服务连接正常")
                return True
            else:
                self.logger.warning(f"Whisper API服务异常，状态码: {response.status_code}")
                print(f"⚠️ Whisper API服务异常，状态码: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Whisper API连接失败: {str(e)}")
            print(f"⚠️ Whisper API连接失败: {str(e)}")
            return False
        
    def process_streamers(self, streamers_config):
        """处理多个主播的数据提取任务"""
        results = {}
        
        total_streamers = len(streamers_config)
        with tqdm(total=total_streamers, desc="主播处理进度", unit="个") as pbar:
            for streamer in streamers_config:
                streamer_id = streamer['id']
                input_dir = streamer['video_dir']
                output_dir = os.path.join(streamer['output_dir'], streamer_id)
                
                min_duration = streamer.get('min_mandarin_minutes', 10) * 60
                max_duration = streamer.get('max_mandarin_minutes', 15) * 60
                
                self.logger.info(f"开始处理主播: {streamer_id}")
                print(f"\n{'='*50}\n开始处理主播: {streamer_id}\n{'='*50}")
                pbar.set_description(f"处理主播: {streamer_id}")
                
                # 创建输出目录
                mandarin_dir = os.path.join(output_dir, "mandarin")
                dialect_dir = os.path.join(output_dir, "dialect")
                os.makedirs(mandarin_dir, exist_ok=True)
                os.makedirs(dialect_dir, exist_ok=True)
                
                # 处理主播视频，直到提取足够的普通话素材
                extraction_result = self.extract_until_sufficient(
                    input_dir, output_dir, streamer_id, min_duration, max_duration)
                
                # 保存元数据
                metadata_path = os.path.join(output_dir, "metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(extraction_result, f, indent=2, ensure_ascii=False)
                
                results[streamer_id] = extraction_result
                
                if extraction_result["sufficient"]:
                    self.logger.info(f"主播 {streamer_id} 素材提取完成！获取了 {extraction_result['mandarin_duration']/60:.2f} 分钟普通话")
                    print(f"✅ 主播 {streamer_id} 素材提取完成！获取了 {extraction_result['mandarin_duration']/60:.2f} 分钟普通话")
                else:
                    self.logger.warning(f"主播 {streamer_id} 素材不足！仅获取了 {extraction_result['mandarin_duration']/60:.2f} 分钟普通话")
                    print(f"⚠️ 警告：主播 {streamer_id} 素材不足！仅获取了 {extraction_result['mandarin_duration']/60:.2f} 分钟普通话")
                
                # 更新总进度条
                pbar.update(1)
        
        # 保存提取报告
        report_path = os.path.join(os.path.dirname(streamers_config[0]['output_dir']), "extraction_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        print(f"\n提取报告已保存至: {report_path}")
        return results
            
    def extract_until_sufficient(self, videos_dir, output_dir, streamer_id, min_duration, max_duration):
        """处理视频直到提取到足够的普通话素材"""
        # 获取所有视频文件
        video_files = self._get_video_files(videos_dir)
        total_videos = len(video_files)
        
        mandarin_dir = os.path.join(output_dir, "mandarin")
        dialect_dir = os.path.join(output_dir, "dialect")             
        
        
        #
                # 初始化结果数据
        result = {
            "streamer_id": streamer_id,
            "total_videos": total_videos,
            "total_extracted_duration": 0,
            "mandarin_clips": 0,
            "mandarin_duration": 0,
            "dialect_clips": 0,
            "dialect_duration": 0,
            "non_target_speaker_clips": 0,  # 添加非目标说话人统计
            "extraction_date": time.strftime("%Y-%m-%d"),
            "sufficient": False,
            "processed_files": []
        }
        
        self.logger.info(f"找到 {total_videos} 个视频文件")
        print(f"\U0001F3A5 找到 {total_videos} 个视频文件")
        
        # 使用tqdm显示视频处理进度
        with tqdm(total=len(video_files), desc="视频处理进度", unit="个") as video_pbar:
            # 处理每个视频文件，直到达到目标时长
            for video_path in video_files:
                if result["mandarin_duration"] >= max_duration:
                    self.logger.info(f"已达到目标提取量 ({max_duration/60:.1f}分钟)，停止处理")
                    print(f"\U0001F3AF 已达到目标提取量 ({max_duration/60:.1f}分钟)，停止处理")
                    result["sufficient"] = True
                    break
                
                video_name = os.path.basename(video_path)
                video_pbar.set_description(f"处理视频: {video_name}")
                
                # 记录处理的文件
                file_result = {
                    "file_name": video_name,
                    "mandarin_clips": 0,
                    "mandarin_duration": 0,
                    "dialect_clips": 0,
                    "dialect_duration": 0,
                    "non_target_clips": 0
                }
                
                # 处理视频提取音频片段
                self.logger.info(f"处理视频: {video_name}")
                temp_dir = os.path.join(output_dir, "temp")
                os.makedirs(temp_dir, exist_ok=True)
                os.chmod(temp_dir, 0o777)  # 设置权限确保可写入
                
                try:
                    # 提取音频片段
                    audio_segments = self.video_processor.process_video(video_path, temp_dir)
                    
                    if not audio_segments:
                        self.logger.warning(f"未从视频中提取到音频片段: {video_name}")
                        video_pbar.update(1)
                        continue
                    
                    # 处理每个音频片段
                    self.logger.info(f"开始处理 {len(audio_segments)} 个音频片段")
                    
                    # 使用多线程处理音频片段
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = []
                        for audio_path in audio_segments:
                            future = executor.submit(self._process_audio_segment, 
                                                   audio_path, 
                                                   streamer_id,
                                                   mandarin_dir, 
                                                   dialect_dir)
                            futures.append(future)
                        
                        # 收集结果
                        with tqdm(total=len(futures), desc="音频分析", unit="段", leave=False) as pbar:
                            for future in futures:
                                segment_result = future.result()
                                if segment_result:
                                    lang_type = segment_result["lang_type"]
                                    duration = segment_result["duration"]
                                    
                                    # 更新统计
                                    if lang_type == "mandarin":
                                        file_result["mandarin_clips"] += 1
                                        file_result["mandarin_duration"] += duration
                                    elif lang_type == "dialect":
                                        file_result["dialect_clips"] += 1
                                        file_result["dialect_duration"] += duration
                                    elif lang_type == "non_target":
                                        file_result["non_target_clips"] += 1
                                
                                pbar.update(1)
                    
                    # 更新总结果
                    result["mandarin_clips"] += file_result["mandarin_clips"]
                    result["mandarin_duration"] += file_result["mandarin_duration"]
                    result["dialect_clips"] += file_result["dialect_clips"]
                    result["dialect_duration"] += file_result["dialect_duration"]
                    result["non_target_speaker_clips"] += file_result["non_target_clips"]
                    result["total_extracted_duration"] += (file_result["mandarin_duration"] + file_result["dialect_duration"])
                    
                    # 添加文件处理结果
                    result["processed_files"].append(file_result)
                    
                    # 显示当前文件处理结果
                    self.logger.info(f"视频 {video_name} 处理完成: 提取 {file_result['mandarin_clips']} 个普通话片段 ({file_result['mandarin_duration']/60:.2f}分钟)")
                    print(f"\U0001F4CA {video_name}: 提取 {file_result['mandarin_clips']} 个普通话片段 ({file_result['mandarin_duration']/60:.2f}分钟)")
                    
                    # 检查是否达到目标
                    if result["mandarin_duration"] >= min_duration:
                        result["sufficient"] = True
                        if result["mandarin_duration"] >= max_duration:
                            self.logger.info(f"已达到目标提取量，停止处理")
                            print(f"\U0001F3AF 已达到目标提取量 ({max_duration/60:.1f}分钟)，停止处理")
                            break
                
                except Exception as e:
                    self.logger.error(f"处理视频 {video_name} 时出错: {str(e)}")
                    print(f"❌ 处理视频 {video_name} 失败: {str(e)}")
                
                finally:
                    # 清理临时文件
                    shutil.rmtree(temp_dir, ignore_errors=True, onerror=self._handle_remove_error)
                    video_pbar.update(1)
        
        # 处理完成，返回结果
        return result
    
    def _handle_remove_error(self, func, path, exc_info):
        """处理文件删除权限错误"""
        try:
            os.chmod(path, 0o777)
            func(path)
        except Exception as e:
            self.logger.error(f"删除文件失败: {path}, 错误: {str(e)}")
    
    def _process_audio_segment(self, audio_path, streamer_id, mandarin_dir, dialect_dir):
        """处理单个音频片段，检测语言并分类"""
        try:
            # 验证音频文件是否有效
            if not self.audio_processor.validate_audio_file(audio_path):
                self.logger.warning(f"无效音频文件: {os.path.basename(audio_path)}")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return None
            
            # 获取音频时长
            duration = self._get_audio_duration(audio_path)
            
            # 如果启用了声纹过滤，先检查是否是目标说话人
            if self.voice_filter:
                is_target, similarity = self.voice_filter.filter_audio(audio_path, streamer_id)
                if not is_target:
                    self.logger.info(f"非目标说话人音频: {os.path.basename(audio_path)}, 相似度: {similarity:.4f}")
                    # 删除非目标说话人的音频
                    os.remove(audio_path)
                    return {
                        "lang_type": "non_target",
                        "duration": 0,
                        "path": None
                    }
            
            # 检测语言
            lang_result = self._detect_language(audio_path)
            
            if not lang_result:
                self.logger.warning(f"语言检测失败: {os.path.basename(audio_path)}")
                os.remove(audio_path)
                return None
            
            lang = lang_result.get("language", "unknown")
            text = lang_result.get("text", "").strip()
            
            # 分类并移动文件
            if lang == "zh" and text:  # 普通话
                # 规范化音频
                normalized_path = self.audio_processor.normalize_audio(audio_path)
                if normalized_path:
                    # 移动到普通话目录
                    dest_path = os.path.join(mandarin_dir, os.path.basename(normalized_path))
                    shutil.move(normalized_path, dest_path)
                    # 删除原始文件
                    if os.path.exists(audio_path) and audio_path != normalized_path:
                        os.remove(audio_path)
                    
                    return {
                        "lang_type": "mandarin",
                        "duration": duration,
                        "path": dest_path,
                        "text": text
                    }
            elif lang in ["zh-yue", "zh-min-nan"] and text:  # 方言
                # 规范化音频
                normalized_path = self.audio_processor.normalize_audio(audio_path)
                if normalized_path:
                    # 移动到方言目录
                    dest_path = os.path.join(dialect_dir, os.path.basename(normalized_path))
                    shutil.move(normalized_path, dest_path)
                    # 删除原始文件
                    if os.path.exists(audio_path) and audio_path != normalized_path:
                        os.remove(audio_path)
                    
                    return {
                        "lang_type": "dialect",
                        "duration": duration,
                        "path": dest_path,
                        "text": text
                    }
            else:
                # 删除不需要的音频
                os.remove(audio_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"处理音频片段失败: {str(e)}")
            # 尝试删除可能损坏的文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None
    
    def _detect_language(self, audio_path):
        """调用Whisper API检测语言并转写"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with open(audio_path, "rb") as f:
                    files = {"audio_file": f}
                    response = requests.post(
                        self.whisper_api_url,
                        files=files,
                        timeout=30  # 添加超时设置
                    )
                
                if response.status_code == 200:
                    try:
                        # 检查响应是否为空
                        if not response.content or len(response.content.strip()) == 0:
                            self.logger.warning(f"API返回空响应，重试 {retry_count+1}/{max_retries}")
                            retry_count += 1
                            time.sleep(1)
                            continue
                            
                        result = response.json()
                        return result
                    except json.JSONDecodeError as e:
                        self.logger.error(f"API返回的JSON格式无效: {response.text[:100]}, 错误: {str(e)}")
                        retry_count += 1
                        time.sleep(1)
                        continue
                else:
                    self.logger.error(f"API请求失败: {response.status_code} - {response.text[:100]}")
                    retry_count += 1
                    time.sleep(1)
                    continue
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"语言检测请求失败: {str(e)}")
                retry_count += 1
                time.sleep(1)
                continue
            except Exception as e:
                self.logger.error(f"语言检测过程中发生未知错误: {str(e)}")
                return None
        
        self.logger.error(f"语言检测失败，已重试 {max_retries} 次")
        return None
    
    def process_batch(self, segment_paths, batch_size=5):
        """分批处理音频片段，避免API过载"""
        results = []
        for i in range(0, len(segment_paths), batch_size):
            batch = segment_paths[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                batch_results = list(executor.map(self._process_audio_segment, batch))
            results.extend(batch_results)
            time.sleep(1)  # 批次间暂停，避免API过载
        return results
    
    def _get_audio_duration(self, audio_path):
        """获取音频文件的时长（秒）"""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # 毫秒转秒
        except Exception as e:
            self.logger.error(f"获取音频时长失败: {str(e)}")
            return 0
    
    def _get_video_files(self, directory):
        """获取目录中的所有视频文件"""
        video_extensions = ['.mp4', '.avi', '.mkv', '.flv', '.mov', '.wmv']
        video_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        
        # 按文件名排序
        video_files.sort()
        return video_files

# 如果直接运行此脚本，执行测试
if __name__ == "__main__":
    print("音频处理模块加载成功")
