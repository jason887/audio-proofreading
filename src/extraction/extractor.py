import os
import time
import json
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import requests

# 导入自定义模块
from .video_processor import VideoProcessor, AudioProcessor

class VoiceDataExtractor:
    def __init__(self, config_path="config/system_config.json", 
                 whisper_api_url=None, min_duration_minutes=10,
                 max_duration_minutes=15, voice_prints_dir=None):
        
        # 初始化配置
        system_config = {}
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    system_config = json.load(f)
            else:
                print(f"⚠️ 系统配置文件不存在: {config_path}，将使用默认配置")
        except Exception as e:
            print(f"⚠️ 加载系统配置文件失败: {str(e)}，将使用默认配置")
            
        self.whisper_api_url = whisper_api_url or system_config.get("whisper_api_url", "http://localhost:9000/asr")
        self.min_duration_seconds = min_duration_minutes * 60
        self.max_duration_seconds = max_duration_minutes * 60
        
        # GPU检测
        try:
            import tensorflow as tf
            self.gpus = tf.config.list_physical_devices('GPU')
            if self.gpus:
                print(f'\n\U0001F680 检测到GPU加速: {len(self.gpus)}个设备可用')
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                print('\n⚠️ 未检测到GPU，将使用CPU处理')
        except Exception as e:
            self.gpus = []
            print(f'\n⚠️ TensorFlow初始化失败: {str(e)}，将使用CPU处理')
        
        self.max_workers = system_config.get("max_workers", 8)
        print(f'\U0001F9F5 多线程处理: {self.max_workers}个工作线程\n')
        
        # 初始化处理器
        self.video_processor = VideoProcessor(
            ffmpeg_path=system_config.get("ffmpeg_path"),
            max_workers=self.max_workers,
            file_size_threshold=system_config.get("file_size_threshold", 1024)
        )
        
        self.audio_processor = AudioProcessor()
        
        # 日志配置
        os.makedirs("logs", exist_ok=True)
        self.logger = logging.getLogger("extractor")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler("logs/extraction.log", encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # 声纹过滤器
        self.voice_filter = None
        if voice_prints_dir and os.path.exists(voice_prints_dir):
            try:
                from .voice_filter_processor import VoiceFilterProcessor
                self.voice_filter = VoiceFilterProcessor(voice_prints_dir)
                self.logger.info(f"声纹过滤器已启用，目录: {voice_prints_dir}")
            except ImportError:
                print("⚠️ 声纹过滤器模块导入失败，将不使用声纹过滤")
                self.logger.warning("声纹过滤器模块导入失败，将不使用声纹过滤")
        else:
            self.logger.info("声纹过滤器未启用")
            
        # 检查API可用性
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
        if not streamers_config:
            self.logger.error("主播配置为空")
            print("错误: 主播配置为空")
            return {}
            
        results = {}
        
        total_streamers = len(streamers_config)
        with tqdm(total=total_streamers, desc="主播处理进度", unit="个") as pbar:
            for streamer in streamers_config:
                if not isinstance(streamer, dict):
                    self.logger.error(f"主播配置格式错误: {streamer}")
                    print(f"错误: 主播配置格式错误: {streamer}")
                    continue
                    
                streamer_id = streamer.get('id')
                if not streamer_id:
                    self.logger.error("主播配置缺少id字段")
                    print("错误: 主播配置缺少id字段")
                    continue
                    
                input_dir = streamer.get('video_dir')
                if not input_dir or not os.path.exists(input_dir):
                    self.logger.error(f"主播 {streamer_id} 的视频目录不存在: {input_dir}")
                    print(f"错误: 主播 {streamer_id} 的视频目录不存在: {input_dir}")
                    continue
                    
                output_dir = streamer.get('output_dir')
                if not output_dir:
                    output_dir = "output"
                    self.logger.warning(f"主播 {streamer_id} 未指定输出目录，使用默认: {output_dir}")
                    
                output_dir = os.path.join(output_dir, streamer_id)
                
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
        if results:
            try:
                report_dir = os.path.dirname(streamers_config[0].get('output_dir', "output"))
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(report_dir, "extraction_report.json")
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(f"\n提取报告已保存至: {report_path}")
            except Exception as e:
                self.logger.error(f"保存提取报告失败: {str(e)}")
                print(f"⚠️ 保存提取报告失败: {str(e)}")
                
        return results

    def extract_until_sufficient(self, videos_dir, output_dir, streamer_id, min_duration, max_duration):
        """处理视频直到提取到足够的普通话素材"""
        # 获取所有视频文件
        video_files = self._get_video_files(videos_dir)
        total_videos = len(video_files)
        
        mandarin_dir = os.path.join(output_dir, "mandarin")
        dialect_dir = os.path.join(output_dir, "dialect")             
        
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
        
        if total_videos == 0:
            self.logger.warning(f"未找到视频文件: {videos_dir}")
            print(f"⚠️ 未找到视频文件: {videos_dir}")
            return result
        
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
