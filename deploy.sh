#!/bin/bash

# 停止当前运行的Flask应用
pkill -f "python app.py"
pkill -f "gunicorn"

# 等待2秒确保进程完全停止
sleep 2

# 更新文件权限
chmod +x app.py

# 重启应用
nohup python app.py > flask.log 2>&1 &

echo "Deployment completed!"
