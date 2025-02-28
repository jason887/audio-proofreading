@echo off
chcp 65001 > nul
echo ==========================================
echo Whisper ASR 批量处理工具启动器
echo ==========================================

:: 使用Anaconda的完整路径
set CONDA_PATH=C:\ProgramData\anaconda3\Scripts\conda.exe
if not exist "%CONDA_PATH%" (
    set CONDA_PATH=C:\Users\%USERNAME%\anaconda3\Scripts\conda.exe
)
if not exist "%CONDA_PATH%" (
    set CONDA_PATH=C:\Anaconda3\Scripts\conda.exe
)
if not exist "%CONDA_PATH%" (
    echo 错误：找不到conda可执行文件，请确认Anaconda安装路径。
    pause
    exit /b 1
)

:: 设置Anaconda根目录
for %%i in ("%CONDA_PATH%") do set CONDA_ROOT=%%~dpi..

:: 使用Anaconda提供的激活脚本
echo 正在激活whisper_client环境...
call "%CONDA_ROOT%\Scripts\activate.bat" whisper_client
if %errorlevel% neq 0 (
    echo 错误：无法激活whisper_client环境！
    echo 正在尝试创建环境...
    
    :: 初始化conda（如果需要的话）
    "%CONDA_PATH%" init cmd.exe
    
    :: 创建环境
    "%CONDA_PATH%" create -n whisper_client python=3.10 -y
    if %errorlevel% neq 0 (
        echo 创建环境失败！
        pause
        exit /b 1
    )
    
    :: 再次尝试激活
    call "%CONDA_ROOT%\Scripts\activate.bat" whisper_client
    if %errorlevel% neq 0 (
        echo 激活环境再次失败！
        pause
        exit /b 1
    )
    
    :: 安装必要的包
    pip install requests tqdm
)

:: 运行Python脚本
echo 正在启动批量处理脚本...
python whisper_process.py
if %errorlevel% neq 0 (
    echo 脚本执行出错！
    pause
    exit /b 1
)

echo 处理完成！
pause
