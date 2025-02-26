# 安装必要的依赖
# pip install aliyun-python-sdk-core oss2 alibabacloud_nls

import os
import json
import time
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
from aliyunsdkcore.request import CommonRequest
import oss2
from alibabacloud_nls.request import FileTransRequest
from alibabacloud_nls import NlsClient

# 阿里云配置
class AliyunConfig:
    ACCESS_KEY_ID = "您的AccessKeyId"  # 替换为您的AccessKeyId
    ACCESS_KEY_SECRET = "您的AccessKeySecret"  # 替换为您的AccessKeySecret
    OSS_BUCKET_NAME = "coze-test-jason"  # 您的OSS桶名
    OSS_ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"  # 您的OSS区域Endpoint
    NLS_APP_KEY = "您的语音识别AppKey"  # 替换为您的语音识别AppKey
    REGION_ID = "cn-shanghai"  # 语音识别服务区域

# 初始化OSS客户端
def init_oss_client():
    auth = oss2.Auth(AliyunConfig.ACCESS_KEY_ID, AliyunConfig.ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, AliyunConfig.OSS_ENDPOINT, AliyunConfig.OSS_BUCKET_NAME)
    return bucket

# 上传文件到OSS
def upload_to_oss(file_path, oss_path):
    bucket = init_oss_client()
    with open(file_path, 'rb') as fileobj:
        bucket.put_object(oss_path, fileobj)
    
    # 生成可访问的URL
    url = f"https://{AliyunConfig.OSS_BUCKET_NAME}.{AliyunConfig.OSS_ENDPOINT}/{oss_path}"
    return url

# 调用阿里云语音识别服务
def speech_to_text(audio_url):
    # 创建AcsClient实例
    client = AcsClient(
        AliyunConfig.ACCESS_KEY_ID,
        AliyunConfig.ACCESS_KEY_SECRET,
        AliyunConfig.REGION_ID
    )
    
    # 创建request，并设置参数
    request = CommonRequest()
    request.set_domain("nls-filetrans.cn-shanghai.aliyuncs.com")
    request.set_method('POST')
    request.set_protocol_type('https')  # 使用HTTPS协议
    request.set_version('2018-08-17')
    request.set_action_name('SubmitTask')
    
    # 设置请求参数
    task = {
        "appkey": AliyunConfig.NLS_APP_KEY,
        "file_link": audio_url,
        "version": "4.0",
        "enable_words": True,
        "enable_sample_rate_adaptive": True
    }
    
    # 将task参数转换为JSON并设置
    task_json = json.dumps(task)
    request.add_body_params('Task', task_json)
    
    try:
        # 发起请求
        response = client.do_action_with_exception(request)
        response_json = json.loads(response)
        
        # 获取任务ID
        task_id = response_json.get("TaskId")
        if not task_id:
            return {"error": "获取任务ID失败"}
        
        # 轮询获取识别结果
        status = "RUNNING"
        result = {}
        
        while status == "RUNNING" or status == "QUEUEING":
            # 创建获取结果的请求
            get_request = CommonRequest()
            get_request.set_domain("nls-filetrans.cn-shanghai.aliyuncs.com")
            get_request.set_method('POST')
            get_request.set_protocol_type('https')  # 使用HTTPS协议
            get_request.set_version('2018-08-17')
            get_request.set_action_name('GetTaskResult')
            get_request.add_body_params('TaskId', task_id)
            
            # 发起请求获取结果
            get_response = client.do_action_with_exception(get_request)
            get_result = json.loads(get_response)
            
            status = get_result.get("Status", "")
            
            if status == "SUCCESS":
                result = get_result
                break
            elif status == "FAILED":
                return {"error": "识别失败", "reason": get_result.get("StatusText", "")}
            
            # 等待一段时间再查询
            time.sleep(3)
        
        # 处理识别结果
        sentences = []
        if "Result" in result and result["Result"]:
            result_detail = json.loads(result["Result"])
            if "Sentences" in result_detail:
                for sentence in result_detail["Sentences"]:
                    sentences.append({
                        "text": sentence.get("Text", ""),
                        "begin_time": sentence.get("BeginTime", 0),
                        "end_time": sentence.get("EndTime", 0)
                    })
        
        return {
            "success": True,
            "sentences": sentences,
            "full_text": " ".join([s["text"] for s in sentences])
        }
        
    except ClientException as e:
        return {"error": f"客户端异常: {str(e)}"}
    except ServerException as e:
        return {"error": f"服务器异常: {str(e)}"}
    except Exception as e:
        return {"error": f"未知异常: {str(e)}"}
