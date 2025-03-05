import os
import sys
# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # training目录
project_root = os.path.dirname(os.path.dirname(current_dir))  # 回退两级到项目根目录
sys.path.insert(0, project_root)
import json
import argparse
from pathlib import Path

# 直接导入本地模块，避免相对导入问题
from src.extraction.extractor import VoiceDataExtractor

def setup_logger(name, log_file, level=None):
    """设置日志记录器"""
    import logging
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def load_config(config_path):
    """加载配置文件并处理可能的错误"""
    try:
        if not os.path.exists(config_path):
            print(f"警告: 配置文件不存在: {config_path}")
            return []
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 确保返回一个列表
        if isinstance(config, dict) and "streamers" in config:
            return config["streamers"]
        elif isinstance(config, list):
            return config
        else:
            print(f"警告: 配置文件格式不正确: {config_path}")
            return []
            
    except json.JSONDecodeError:
        print(f"错误: 配置文件JSON格式无效: {config_path}")
        return []
    except Exception as e:
        print(f"错误: 加载配置文件失败: {str(e)}")
        return []

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='声音克隆流水线工具')
    parser.add_argument('--config', default='config/streamers.json', help='主播配置文件路径')
    parser.add_argument('--system_config', default='config/system_config.json', help='系统配置文件路径')
    parser.add_argument('--min_mandarin', type=int, default=10, help='最小普通话素材时长(分钟)')
    parser.add_argument('--max_mandarin', type=int, default=15, help='最大普通话素材时长(分钟)')
    parser.add_argument('--voice_prints', default=None, help='声纹文件目录路径')
    
    args = parser.parse_args()
    
    # 确保配置目录存在
    os.makedirs("config", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 设置主日志
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = setup_logger("main", "logs/main.log", logging.INFO)
    
    # 显示启动信息
    print("\n" + "="*60)
    print(" 声音克隆流水线工具 ".center(58, "="))
    print("="*60)
    
    # 加载配置
    try:
        streamers = load_config(args.config)
        if not streamers:
            logger.error(f"主播配置文件加载失败或为空: {args.config}")
            print(f"错误: 主播配置文件加载失败或为空: {args.config}")
            sys.exit(1)
            
        logger.info(f"加载了 {len(streamers)} 个主播配置")
        print(f"加载了 {len(streamers)} 个主播配置")
        
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}")
        print(f"错误: 加载配置失败: {str(e)}")
        sys.exit(1)
    
    try:
        # 初始化提取器
        extractor = VoiceDataExtractor(
            config_path=args.system_config,
            min_duration_minutes=args.min_mandarin,
            max_duration_minutes=args.max_mandarin,
            voice_prints_dir=args.voice_prints  # 添加声纹目录参数
        )
        
        # 处理所有配置的主播
        results = extractor.process_streamers(streamers)
        
        if not results:
            logger.error("处理结果为空")
            print("错误: 处理结果为空")
            sys.exit(1)
        
        # 显示处理结果
        print("\n" + "="*60)
        print(" 处理完成 ".center(58, "="))
        print("="*60)
        
        for streamer_id, result in results.items():
            status = "✅ 成功" if result["sufficient"] else "⚠️ 素材不足"
            print(f"{status} {streamer_id}: {result['mandarin_duration']/60:.2f}分钟")
        
        if args.voice_prints:
            print("\n声纹过滤统计:")
            for streamer_id, result in results.items():
                if "non_target_speaker_clips" in result and result["non_target_speaker_clips"] > 0:
                    print(f"  - {streamer_id}: 过滤了 {result['non_target_speaker_clips']} 个非主播音频片段")
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        print(f"错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
