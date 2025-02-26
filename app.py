from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import soundfile as sf
import numpy as np
import os
import logging
from datetime import datetime
import json
import secrets
import hashlib
from functools import wraps

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保必要的目录存在
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
for folder in [UPLOAD_FOLDER, DATA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 用户数据文件
USERS_FILE = os.path.join(DATA_FOLDER, 'users.json')
CORRECTIONS_FILE = os.path.join(DATA_FOLDER, 'corrections.json')

# 初始化用户数据文件
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump([], f)  # 初始化为空列表而不是字典

# 初始化校对数据文件
if not os.path.exists(CORRECTIONS_FILE):
    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump([], f)

# Token验证装饰器
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': '无效的token'}), 401

        if not token:
            return jsonify({'message': '缺少token'}), 401

        try:
            # 这里可以添加token验证逻辑
            # 简单示例中我们只检查token是否存在
            pass
        except:
            return jsonify({'message': '无效的token'}), 401

        return f(*args, **kwargs)
    return decorated

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 注册接口
@app.route('/api/register', methods=['POST'])
def register():
    try:
        # 添加请求数据的日志记录
        logger.info(f"Received registration request. Content-Type: {request.content_type}")
        logger.info(f"Request data: {request.get_data(as_text=True)}")
        
        # 确保数据是JSON格式
        if not request.is_json:
            return jsonify({'message': '请求必须是JSON格式'}), 400

        data = request.get_json()
        
        # 记录解析后的JSON数据
        logger.info(f"Parsed JSON data: {data}")

        if not isinstance(data, dict):
            return jsonify({'message': '无效的数据格式'}), 400

        username = data.get('username')
        password = data.get('password')

        # 验证数据
        if not username or not isinstance(username, str):
            return jsonify({'message': '用户名无效'}), 400
        if not password or not isinstance(password, str):
            return jsonify({'message': '密码无效'}), 400

        with open(USERS_FILE, 'r') as f:
            users = json.load(f)

        # 检查用户名是否已存在
        if any(user['username'] == username for user in users):
            return jsonify({'message': '用户名已存在'}), 400

        # 密码加密存储
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        new_user = {
            'username': username,
            'password': hashed_password,
            'created_at': datetime.now().isoformat()
        }
        users.append(new_user)

        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)

        return jsonify({'message': '注册成功'}), 201

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'message': f'注册过程中发生错误: {str(e)}'}), 500

# 登录接口
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    with open(USERS_FILE, 'r') as f:
        users = json.load(f)

    # 查找用户
    user = next((user for user in users if user['username'] == username), None)
    if not user:
        return jsonify({'message': '用户名或密码错误'}), 401

    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    if user['password'] != hashed_password:
        return jsonify({'message': '用户名或密码错误'}), 401

    # 生成token
    token = secrets.token_hex(32)
    return jsonify({
        'token': token,
        'username': username
    })

# 获取音频列表
@app.route('/api/audio-list')
@token_required
def get_audio_list():
    # 从文件读取校对数据
    with open(CORRECTIONS_FILE, 'r') as f:
        corrections = json.load(f)
    return jsonify(corrections)

# 提交校对
@app.route('/api/submit-correction', methods=['POST'])
@token_required
def submit_correction():
    data = request.json
    with open(CORRECTIONS_FILE, 'r') as f:
        corrections = json.load(f)

    # 更新校对数据
    for item in corrections:
        if item['id'] == data['id']:
            item['correctedText'] = data['correctedText']
            item['corrector'] = data['corrector']
            item['timestamp'] = data['timestamp']
            break

    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump(corrections, f)

    return jsonify({'message': '更正已保存'})

# 导出数据
@app.route('/api/export')
@token_required
def export_data():
    with open(CORRECTIONS_FILE, 'r') as f:
        corrections = json.load(f)
    
    response = jsonify(corrections)
    response.headers.set('Content-Disposition', 'attachment; filename=correction_data.json')
    return response

# 音频分析接口
@app.route('/api/analyze', methods=['POST'])
@token_required
def analyze_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 生成唯一的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 保存上传的文件
        file.save(filepath)
        logger.info(f"File saved to {filepath}")

        try:
            # 使用 soundfile 加载音频文件
            audio_data, sample_rate = sf.read(filepath)
            
            # 获取音频信息
            duration_seconds = len(audio_data) / sample_rate
            channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
            frame_rate = sample_rate
            
            # 计算最大振幅
            max_amplitude = float(np.max(np.abs(audio_data)))
            
            # 计算RMS值作为响度的替代
            rms = np.sqrt(np.mean(np.square(audio_data)))
            # 将RMS转换为dB
            loudness = 20 * np.log10(rms) if rms > 0 else -96.0

            # 准备返回数据
            audio_info = {
                'filename': file.filename,
                'duration': duration_seconds,
                'channels': channels,
                'frame_rate': frame_rate,
                'sample_width': audio_data.dtype.itemsize,
                'max_amplitude': max_amplitude,
                'loudness_dbfs': loudness
            }

            return jsonify(audio_info)

        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            return jsonify({'error': f'Error processing audio file: {str(e)}'}), 500

        finally:
            # 清理上传的文件
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# 健康检查接口
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
