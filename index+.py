<!-- 在index.html中添加以下内容 -->

<div class="upload-section" style="margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
    <h2>上传音频进行识别</h2>
    <form id="upload-form" enctype="multipart/form-data">
        <input type="file" id="audio-file" name="file" accept="audio/*" style="margin-bottom: 10px;">
        <button type="submit" class="submit-correction" style="margin-left: 10px;">上传并识别</button>
    </form>
    <div id="upload-status" style="margin-top: 10px;"></div>
    <div class="progress-container" style="margin-top: 10px; display: none;">
        <div class="progress-bar" style="height: 20px; background-color: #f0f0f0; border-radius: 4px; overflow: hidden;">
            <div class="progress" style="height: 100%; width: 0%; background-color: #2196F3; transition: width 0.3s;"></div>
        </div>
        <div class="progress-text" style="margin-top: 5px; text-align: center;">0%</div>
    </div>
</div>

<!-- 添加上传和识别的JavaScript代码 -->
<script>
    // 添加到现有的script标签中
    
    // 处理音频上传和识别
    document.getElementById('upload-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // 检查登录状态
        if (!checkLoginStatus()) {
            alert('请先登录');
            return;
        }
        
        const fileInput = document.getElementById('audio-file');
        if (!fileInput.files.length) {
            alert('请选择音频文件');
            return;
        }
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        // 显示上传状态
        const uploadStatus = document.getElementById('upload-status');
        uploadStatus.textContent = '正在上传并识别音频，请稍候...';
        
        // 显示进度条
        const progressContainer = document.querySelector('.progress-container');
        const progressBar = document.querySelector('.progress');
        const progressText = document.querySelector('.progress-text');
        progressContainer.style.display = 'block';
        
        try {
            // 获取token
            const token = localStorage.getItem('userToken');
            
            const response = await fetch('/api/recognize-audio', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                uploadStatus.textContent = '音频识别成功！';
                progressBar.style.width = '100%';
                progressText.textContent = '100%';
                
                // 刷新音频列表
                loadAudioList();
                
                // 3秒后隐藏进度条
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    uploadStatus.textContent = '';
                }, 3000);
            } else {
                const error = await response.json();
                uploadStatus.textContent = `识别失败: ${error.error || '未知错误'}`;
                progressBar.style.backgroundColor = '#f44336';
            }
        } catch (error) {
            uploadStatus.textContent = `发生错误: ${error.message}`;
            progressBar.style.backgroundColor = '#f44336';
        }
    });
    
    // 模拟上传进度
    function simulateProgress() {
        const progressBar = document.querySelector('.progress');
        const progressText = document.querySelector('.progress-text');
        let width = 0;
        
        const interval = setInterval(() => {
            if (width >= 90) {
                clearInterval(interval);
                return;
            }
            
            width += Math.random() * 10;
            if (width > 90) width = 90;
            
            progressBar.style.width = `${width}%`;
            progressText.textContent = `${Math.round(width)}%`;
        }, 500);
    }
</script>
