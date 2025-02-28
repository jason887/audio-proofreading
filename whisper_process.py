import requests
import os
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

# 尝试导入依赖，如果不存在则提示安装
try:
    from tqdm import tqdm
except ImportError:
    print("缺少必要的依赖库。正在尝试安装...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "requests"])
        from tqdm import tqdm
        print("依赖安装成功！")
    except Exception as e:
        print(f"无法安装依赖: {e}")
        print("请手动运行: pip install tqdm requests")
        sys.exit(1)

# 配置
BASE_URL = "http://localhost:9000"  # 基础URL
API_URL = f"{BASE_URL}/asr"         # 修正：使用正确的ASR端点（移除v1前缀）
# 修复路径格式问题
AUDIO_DIR = "D:\\my_video_project\\output"  # 使用正确的Windows路径格式
OUTPUT_DIR = "D:\\my_video_project\\transcripts"
MAX_WORKERS = 2  # 并行处理数量（不要太高，避免GPU过载）

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_api_available():
    """检查Whisper API服务是否可用"""
    available = False
    message = ""
    
    # 修正：尝试更合适的健康检查端点
    health_endpoints = ["/openapi.json", "/", "/health"]
    
    for endpoint in health_endpoints:
        try:
            url = f"{BASE_URL}{endpoint}"
            print(f"尝试连接到API端点: {url}")
            response = requests.get(url, timeout=5)
            print(f"状态码: {response.status_code}, 响应: {response.text[:100]}")
            
            if response.status_code == 200:
                available = True
                message = f"API服务正常响应，端点: {endpoint}"
                break
        except Exception as e:
            print(f"尝试连接 {endpoint} 失败: {e}")
    
    if not available:
        # 最后尝试直接访问ASR端点
        try:
            response = requests.post(API_URL, timeout=2)
            # 即使返回错误也表明服务在运行
            available = True
            message = "API服务可能正常，但无健康检查端点"
        except Exception as e:
            message = f"无法连接到API服务: {e}"
    
    return available, message

def process_audio(audio_file):
    """处理单个音频文件"""
    filename = os.path.basename(audio_file)
    try:
        output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json")
        
        # 如果结果已存在，跳过处理
        if os.path.exists(output_file):
            print(f"跳过已处理的文件: {filename}")
            return {"file": filename, "status": "skipped"}
        
        print(f"正在处理: {filename}")
        with open(audio_file, "rb") as f:
            files = {"audio_file": (filename, f, "audio/wav")}
            data = {}
            response = requests.post(API_URL, files=files, data=data)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            # 尝试解析JSON，如果失败则作为纯文本处理
            try:
                result = response.json()
                print(f"响应内容预览: {str(result)[:200]}")
            except json.JSONDecodeError:
                # 纯文本响应处理
                text = response.text.strip()
                print(f"响应内容预览: {text[:200]}")
                result = {"text": text}
            
            # 保存结果
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            
            return {"file": filename, "status": "success"}
        else:
            print(f"API请求失败: {filename}, 状态码: {response.status_code}")
            return {"file": filename, "status": "failed", "code": response.status_code}
    
    except requests.RequestException as e:
        print(f"网络错误: {filename}, {str(e)}")
        return {"file": filename, "status": "error", "message": f"网络错误: {str(e)}"}
    except Exception as e:
        print(f"处理错误: {filename}, {str(e)}")
        return {"file": filename, "status": "error", "message": str(e)}

def batch_process():
    """批量处理所有WAV文件"""
    # 检查音频目录是否存在
    if not os.path.exists(AUDIO_DIR):
        print(f"错误: 音频目录不存在 - {AUDIO_DIR}")
        return False

    # 获取所有WAV文件
    audio_files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) 
                  if f.lower().endswith('.wav') and os.path.getsize(os.path.join(AUDIO_DIR, f)) > 0]
    
    if not audio_files:
        print(f"未找到任何WAV音频文件在 {AUDIO_DIR}")
        return False
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    results = []
    start_time = time.time()
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_audio, audio) for audio in audio_files]
        
        for future in tqdm(futures, desc="处理进度"):
            result = future.result()
            results.append(result)
    
    # 统计结果
    success = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "failed"])
    skipped = len([r for r in results if r["status"] == "skipped"])
    errors = len([r for r in results if r["status"] == "error"])
    
    print(f"\n处理完成!")
    print(f"成功: {success}, 失败: {failed}, 跳过: {skipped}, 错误: {errors}")
    return True

# 合并所有结果为一个文本文件
def merge_results():
    """将所有转写结果合并到一个文本文件中"""
    json_files = []
    
    # 获取所有JSON文件
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.json'):
            try:
                # 尝试提取数字部分用于排序
                parts = os.path.splitext(f)[0].split('_')
                for part in reversed(parts):
                    if part.isdigit():
                        sort_key = int(part)
                        break
                else:
                    # 如果没有找到数字部分，使用文件名作为排序键
                    sort_key = f
                json_files.append((f, sort_key))
            except:
                json_files.append((f, f))  # 使用文件名作为默认排序键
    
    # 按数字或文件名排序
    json_files.sort(key=lambda x: x[1])
    
    all_text = []
    for json_file, _ in json_files:
        try:
            with open(os.path.join(OUTPUT_DIR, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get('text', '').strip()
                if text:
                    all_text.append(text)
        except Exception as e:
            print(f"无法读取文件: {json_file}, 原因: {str(e)}")
    
    if not all_text:
        print("没有找到任何有效的转写结果")
        return
    
    # 保存合并结果
    output_file = os.path.join(OUTPUT_DIR, "完整转写.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_text))
    
    print(f"所有结果已合并到: {output_file}")
    print(f"共合并了 {len(all_text)} 个转写结果")

if __name__ == "__main__":
    print("="*50)
    print("Whisper ASR 批量处理工具")
    print("="*50)
    
    # 检查API是否可用
    available, message = check_api_available()
    if not available:
        print(f"错误：无法连接到Whisper API服务。{message}")
        print(f"请确认服务是否正在运行，且端口映射正确: {BASE_URL}")
        sys.exit(1)
    
    print(f"API服务检查: {message}")
    print(f"将使用API端点: {API_URL}")
    
    # 执行批处理
    if batch_process():
        merge_results()
    else:
        print("由于错误，未能完成处理")
