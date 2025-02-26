# 添加到您现有的app.py中

@app.route('/api/recognize-audio', methods=['POST'])
@token_required
def recognize_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件部分'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        # 生成唯一的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 保存上传的文件
        file.save(filepath)
        logger.info(f"文件保存到 {filepath}")

        try:
            # 上传到OSS
            oss_path = f"audio/{filename}"
            audio_url = upload_to_oss(filepath, oss_path)
            logger.info(f"上传成功,URL: {audio_url}")
            
            # 调用语音识别
            recognition_result = speech_to_text(audio_url)
            
            if "error" in recognition_result:
                return jsonify({'error': f"识别失败: {recognition_result['error']}"}), 500
            
            # 保存识别结果到校对数据中
            with open(CORRECTIONS_FILE, 'r') as f:
                corrections = json.load(f)
            
            # 创建新的校对条目
            new_correction = {
                'id': str(uuid.uuid4()),
                'filename': file.filename,
                'audioUrl': audio_url,
                'originalText': recognition_result['full_text'],
                'correctedText': '',
                'corrector': '',
                'timestamp': '',
                'sentences': recognition_result['sentences']
            }
            
            corrections.append(new_correction)
            
            with open(CORRECTIONS_FILE, 'w') as f:
                json.dump(corrections, f)
            
            return jsonify({
                'success': True,
                'message': '音频上传并识别成功',
                'result': new_correction
            })
            
        except Exception as e:
            logger.error(f"处理音频文件时出错: {str(e)}")
            return jsonify({'error': f'处理音频文件时出错: {str(e)}'}), 500
        
        finally:
            # 清理上传的临时文件
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        logger.error(f"服务器错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500
