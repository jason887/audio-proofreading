import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import shutil
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

class VoiceprintExtractor:
    def __init__(self, output_dir):
        """初始化声纹提取器"""
        self.output_dir = output_dir
        # 检查是否有可用的GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"使用GPU: {gpus}")
            except RuntimeError as e:
                print(f"GPU设置错误: {e}")
        else:
            print("未检测到GPU，将使用CPU进行计算")
            
        # 加载声纹编码器模型 (使用VGG16的部分层作为特征提取器)
        self.encoder = self._create_voiceprint_model()
        print("声纹提取器初始化完成")
    
    def _create_voiceprint_model(self):
        """创建声纹提取模型"""
        # 使用预训练的VGG16作为基础模型
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # 冻结基础模型的层
        for layer in base_model.layers:
            layer.trainable = False
        
        # 添加自定义层用于声纹提取
        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        
        # 创建模型
        model = Model(inputs=base_model.input, outputs=x)
        return model
    
    def _spectrogram_from_audio(self, audio, sr):
        """从音频生成声谱图，用于输入到模型"""
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        # 转换为分贝单位
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # 归一化到[0,1]范围
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        # 调整大小以适应模型输入
        height, width = mel_spec_norm.shape
        if width < 224:
            # 如果宽度不足，则填充
            pad_width = 224 - width
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # 如果宽度过大，则截取
            mel_spec_norm = mel_spec_norm[:, :224]
        
        # 确保高度为224
        if height != 224:
            mel_spec_norm = librosa.util.fix_length(mel_spec_norm, size=224, axis=0)
        
        # 扩展为3通道以适应VGG16输入
        mel_spec_3channel = np.stack([mel_spec_norm, mel_spec_norm, mel_spec_norm], axis=-1)
        return mel_spec_3channel
        
    def extract_audio_from_video(self, video_path, temp_dir="temp_audio"):
        """从视频文件中提取音频"""
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        filename = os.path.basename(video_path).split('.')[0]
        audio_path = os.path.join(temp_dir, f"{filename}.wav")
        
        # 使用pydub从视频中提取音频
        try:
            if video_path.endswith('.mp3'):
                audio = AudioSegment.from_mp3(video_path)
            else:
                audio = AudioSegment.from_file(video_path)
            audio.export(audio_path, format="wav")
            print(f"从 {video_path} 提取音频成功")
            return audio_path
        except Exception as e:
            print(f"音频提取失败: {e}")
            return None
    
    def voice_activity_detection(self, audio_path):
        """检测音频中的人声部分"""
        # 加载音频
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 使用librosa进行语音活动检测
        intervals = librosa.effects.split(y, top_db=20)
        
        # 合并相近的间隔
        merged_intervals = []
        if len(intervals) > 0:
            current_start, current_end = intervals[0]
            for start, end in intervals[1:]:
                # 如果间隔小于0.5秒，则合并
                if start - current_end < 0.5 * sr:
                    current_end = end
                else:
                    merged_intervals.append((current_start, current_end))
                    current_start, current_end = start, end
            merged_intervals.append((current_start, current_end))
        
        # 只保留长度超过1秒的语音段
        valid_intervals = [(start, end) for start, end in merged_intervals if (end - start) > sr]
        
        return y, sr, valid_intervals
    
    def extract_voiceprint(self, audio_path, streamer_name):
        """提取声纹并保存"""
        print(f"正在处理 {streamer_name} 的音频...")
        
        # 创建主播的声纹目录
        streamer_dir = os.path.join(self.output_dir, streamer_name)
        if not os.path.exists(streamer_dir):
            os.makedirs(streamer_dir)
        
        # 检测人声部分
        y, sr, voice_intervals = self.voice_activity_detection(audio_path)
        
        if len(voice_intervals) == 0:
            print(f"未在 {audio_path} 中检测到有效人声")
            return False
        
        # 处理每个语音段
        voiceprints = []
        for i, (start, end) in enumerate(tqdm(voice_intervals, desc="提取声纹")):
            # 提取语音段
            segment = y[start:end]
            
            # 保存语音段
            segment_path = os.path.join(streamer_dir, f"segment_{i}.wav")
            sf.write(segment_path, segment, sr)
            
            # 生成声谱图
            spectrogram = self._spectrogram_from_audio(segment, sr)
            
            # 使用模型提取声纹特征
            spectrogram_batch = np.expand_dims(spectrogram, axis=0)
            embedding = self.encoder.predict(spectrogram_batch, verbose=0)[0]
            
            # 归一化嵌入向量
            embedding = embedding / np.linalg.norm(embedding)
            
            voiceprints.append(embedding)
            
            # 保存声纹数据
            np.save(os.path.join(streamer_dir, f"voiceprint_{i}.npy"), embedding)
        
        # 计算平均声纹
        if voiceprints:
            avg_voiceprint = np.mean(np.array(voiceprints), axis=0)
            # 再次归一化
            avg_voiceprint = avg_voiceprint / np.linalg.norm(avg_voiceprint)
            np.save(os.path.join(streamer_dir, "average_voiceprint.npy"), avg_voiceprint)
            
            print(f"成功提取 {streamer_name} 的声纹，共 {len(voiceprints)} 个语音段")
            return True
        return False
    
    def cleanup(self, temp_dir="temp_audio"):
        """清理临时文件"""
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("临时文件已清理")

def main():
    parser = argparse.ArgumentParser(description="从音频/视频中提取主播声纹")
    parser.add_argument("--input", required=True, help="输入文件路径(MP3或视频)")
    parser.add_argument("--name", required=True, help="主播名称")
    parser.add_argument("--output", default="D:/audio-proofreading/data/voice_prints", 
                        help="声纹输出目录")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return
    
    # 初始化声纹提取器
    extractor = VoiceprintExtractor(args.output)
    
    try:
        # 从视频中提取音频
        audio_path = extractor.extract_audio_from_video(args.input)
        if audio_path:
            # 提取声纹
            success = extractor.extract_voiceprint(audio_path, args.name)
            if success:
                print(f"声纹提取完成，已保存到 {os.path.join(args.output, args.name)}")
            else:
                print("声纹提取失败")
    finally:
        # 清理临时文件
        extractor.cleanup()

if __name__ == "__main__":
    main()
