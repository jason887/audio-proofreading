import os
import logging
import numpy as np
import tensorflow as tf
import librosa

class VoiceFilterProcessor:
    """
    声纹过滤处理器
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
            if not os.path.exists(voice_prints_dir):
                self.logger.error(f'声纹目录不存在: {voice_prints_dir}')
                return
                
            streamer_folders = [f for f in os.listdir(voice_prints_dir) 
                              if os.path.isdir(os.path.join(voice_prints_dir, f))]
            
            if not streamer_folders:
                self.logger.warning(f'声纹目录为空: {voice_prints_dir}')
                return
                
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
