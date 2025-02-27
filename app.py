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
import uuid
from functools import wraps
import oss2
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
from aliyunsdkcore.request import CommonRequest
import time
import base64
import requests  # 新增：用于调用Whisper API [[1]](#__1)

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

# 阿里云配置
ACCESS_KEY_ID = 'YOUR_ACCESS_KEY_ID'
ACCESS_KEY_SECRET = 'YOUR_ACCESS_KEY_SECRET'
OSS_ENDPOINT = 'oss-cn-shenzhen.aliyuncs.com'  # 根据您的OSS区域修改
OSS_BUCKET_NAME = 'coze-test-jason'
ISI_REGION = 'cn-shanghai'  # 智能语音交互服务区域

# Whisper ASR配置 [[2]](#__2)
WHISPER_API_URL = "http://localhost:9000/asr"  # 本地Whisper服务地址，根据实际部署修改
USE_WHISPER = True  # 设置为True使用Whisper，False使用阿里云语音识别

# 初始化OSS客户端
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)

# 初始化智能语音交互客户端
isi_client = AcsClient(ACCESS_KEY_ID, ACCESS_KEY_SECRET, ISI_REGION)

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

# 上传文件到OSS
def upload_to_oss(local_file_path, oss_path):
    try:
        # 上传文件到OSS
        result = bucket.put_object_from_file(oss_path, local_file_path)
        # 生成可访问的URL
        url = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/{oss_path}"
        logger.info(f"文件上传成功: {url}")
        return url
    except Exception as e:
        logger.error(f"上传文件到OSS失败: {str(e)}")
        raise e

# Whisper语音识别函数 [[3]](#__3)
def whisper_speech_to_text(audio_url):
    try:
        # 调用Whisper API
        payload = {"audio_url": audio_url}
        response = requests.post(WHISPER_API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return {
                'full_text': result.get('text', ''),
                'sentences': result.get('segments', [])
            }
        else:
            logger.error(f"Whisper API调用失败: {response.text}")
            return {"error": f"识别失败: {response.text}"}
    except Exception as e:
        logger.error(f"Whisper语音识别出错: {str(e)}")
        return {"error": str(e)}

# 阿里云语音识别函数
def speech_to_text(audio_url):
    try:
        # 创建录音文件识别请求
        request = CommonRequest()
        request.set_domain('filetrans.cn-shanghai.aliyuncs.com')
        request.set_version('2018-08-17')
        request.set_product('nls-filetrans')
        request.set_action_name('SubmitTask')
        request.set_method('POST')
        
        # 设置录音文件识别参数
        task = {
            "appkey": "YOUR_ISI_APPKEY",  # 您的智能语音交互应用的AppKey
            "file_link": audio_url,
            "version": "4.0",
            "enable_words": True,
            "enable_sample_rate_adaptive": True
        }
        
        task_str = json.dumps(task)
        request.add_body_params('Task', task_str)
        
        # 提交录音文件识别请求
        response = isi_client.do_action_with_exception(request)
        response_json = json.loads(response.decode('utf-8'))
        
        if 'TaskId' not in response_json:
            return {"error": "提交识别任务失败"}
        
        task_id = response_json['TaskId']
        logger.info(f"识别任务提交成功，任务ID: {task_id}")
        
        # 轮询获取识别结果
        max_retry = 10
        retry_count = 0
        while retry_count < max_retry:
            time.sleep(5)  # 等待5秒再查询结果
            
            get_result_request = CommonRequest()
            get_result_request.set_domain('filetrans.cn-shanghai.aliyuncs.com')
            get_result_request.set_version('2018-08-17')
            get_result_request.set_product('nls-filetrans')
            get_result_request.set_action_name('GetTaskResult')
            get_result_request.set_method('POST')
            get_result_request.add_body_params('TaskId', task_id)
            
            result_response = isi_client.do_action_with_exception(get_result_request)
            result_json = json.loads(result_response.decode('utf-8'))
            
            status = result_json.get('StatusText')
            if status == 'SUCCESS':
                # 识别成功，返回结果
                result = result_json.get('Result', {})
                sentences = []
                full_text = ""
                
                # 解析识别结果
                for sentence in result.get('Sentences', []):
                    sentences.append({
                        'text': sentence.get('Text', ''),
                        'begin_time': sentence.get('BeginTime', 0),
                        'end_time': sentence.get('EndTime', 0)
                    })
                    full_text += sentence.get('Text', '') + " "
                
                return {
                    'full_text': full_text.strip(),
                    'sentences': sentences
                }
            elif status == 'RUNNING' or status == 'QUEUEING':
                # 任务仍在进行中，继续等待
                retry_count += 1
                continue
            else:
                # 识别失败
                return {"error": f"识别失败，状态: {status}"}
        
        return {"error": "识别超时，请稍后重试"}
    
    except Exception as e:
        logger.error(f"语音识别出错: {str(e)}")
        return {"error": str(e)}

# 获取OSS中的音频文件列表
def get_oss_audio_files(prefix="audio/"):
    try:
        audio_files = []
        for obj in oss2.ObjectIterator(bucket, prefix=prefix):
            if obj.key.endswith(('.wav', '.mp3', '.ogg')):
                url = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/{obj.key}"
                audio_files.append({
                    'key': obj.key,
                    'url': url,
                    'size': obj.size,
                    'last_modified': obj.last_modified
                })
        return audio_files
    except Exception as e:
        logger.error(f"获取OSS音频文件列表失败: {str(e)}")
        return []

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
        logger.info(f"File saved to
