@echo off
chcp 65001 >nul
title Self AGI 完整部署启动脚本 - 版本 1.0
color 0A

echo ========================================================
echo          Self AGI 完整部署启动脚本
echo ========================================================
echo 系统概述: 完整的自主通用人工智能系统
echo 版本: v1.3 (安全与功能全面增强版)
echo 最后更新: 2026年3月29日
echo 许可证: Apache License 2.0
echo 邮箱: silenceceowtom@qq.com
echo ========================================================
echo.

REM ========================================================
REM 第一阶段: 系统环境检查和依赖验证
REM ========================================================
echo [阶段1] 系统环境检查和依赖验证
echo.

REM 检查管理员权限（某些操作需要）
echo [INFO] 检查管理员权限...
net session >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] 当前用户具有管理员权限
    set ADMIN_PRIVILEGES=1
) else (
    echo [WARNING] 当前用户无管理员权限，某些操作可能受限
    set ADMIN_PRIVILEGES=0
)

echo.

REM 检查Python版本
echo [INFO] 检查Python版本...
python --version >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo [SUCCESS] Python版本: %PYTHON_VERSION%
    set PYTHON_CMD=python
) else (
    python3 --version >nul 2>nul
    if %errorlevel% equ 0 (
        for /f "tokens=2" %%i in ('python3 --version') do set PYTHON_VERSION=%%i
        echo [SUCCESS] Python版本: %PYTHON_VERSION%
        set PYTHON_CMD=python3
    ) else (
        echo [ERROR] 未找到Python，请安装Python 3.9+
        echo [INFO] 下载链接: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

REM 验证Python版本号
echo [INFO] 验证Python版本要求（需要3.9+）...
%PYTHON_CMD% -c "import sys; version=sys.version_info; print(f'{version.major}.{version.minor}')" > python_version.txt 2>nul
set /p PYTHON_MAJOR_MINOR=<python_version.txt
del python_version.txt

for /f "tokens=1,2 delims=." %%a in ("%PYTHON_MAJOR_MINOR%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    echo [ERROR] Python版本不符合要求，需要Python 3.9+
    pause
    exit /b 1
) else if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 9 (
        echo [ERROR] Python版本不符合要求，需要Python 3.9+
        pause
        exit /b 1
    )
)
echo [SUCCESS] Python版本检查通过

echo.

REM 检查Node.js版本
echo [INFO] 检查Node.js版本...
node --version >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=1" %%i in ('node --version') do set NODE_VERSION=%%i
    echo [SUCCESS] Node.js版本: %NODE_VERSION%
) else (
    echo [ERROR] 未找到Node.js，请安装Node.js 18+
    echo [INFO] 下载链接: https://nodejs.org/
    pause
    exit /b 1
)

REM 验证Node.js版本
echo [INFO] 验证Node.js版本要求（需要18+）...
node -e "const v = process.version.match(/v(\d+)\./)[1]; console.log(v);" > node_version.txt 2>nul
set /p NODE_MAJOR=<node_version.txt
del node_version.txt

if %NODE_MAJOR% LSS 18 (
    echo [ERROR] Node.js版本不符合要求，需要Node.js 18+
    pause
    exit /b 1
)
echo [SUCCESS] Node.js版本检查通过

echo.

REM 检查npm版本
echo [INFO] 检查npm版本...
npm --version >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=1" %%i in ('npm --version') do set NPM_VERSION=%%i
    echo [SUCCESS] npm版本: %NPM_VERSION%
) else (
    echo [ERROR] 未找到npm，请确保Node.js安装正确
    pause
    exit /b 1
)

echo.

REM 检查CUDA（可选）
echo [INFO] 检查CUDA支持（GPU训练和推理）...
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] 检测到NVIDIA GPU，检查CUDA版本...
    nvidia-smi --query-gpu=driver_version --format=csv,noheader > cuda_driver.txt 2>nul
    set /p CUDA_DRIVER_VERSION=<cuda_driver.txt
    del cuda_driver.txt
    echo [SUCCESS] NVIDIA驱动版本: %CUDA_DRIVER_VERSION%
    
    %PYTHON_CMD% -c "import torch; print(torch.cuda.is_available())" > cuda_available.txt 2>nul
    set /p CUDA_AVAILABLE=<cuda_available.txt
    del cuda_available.txt
    
    if "%CUDA_AVAILABLE%"=="True" (
        echo [SUCCESS] PyTorch CUDA支持已启用
        set GPU_ENABLED=1
    ) else (
        echo [WARNING] PyTorch CUDA支持未启用，将使用CPU模式
        set GPU_ENABLED=0
    )
) else (
    echo [INFO] 未检测到NVIDIA GPU或nvidia-smi不可用，将使用CPU模式
    set GPU_ENABLED=0
)

echo.

REM 检查系统资源
echo [INFO] 检查系统资源...
systeminfo | findstr /C:"可用物理内存" > mem_info.txt 2>nul
set /p MEM_INFO=<mem_info.txt
del mem_info.txt
echo [INFO] 系统内存: %MEM_INFO%

wmic cpu get name | findstr /v "Name" > cpu_info.txt 2>nul
set /p CPU_INFO=<cpu_info.txt
del cpu_info.txt
echo [INFO] CPU信息: %CPU_INFO%

echo.

REM 检查必要目录
echo [INFO] 检查项目目录结构...
if not exist "backend" (
    echo [ERROR] 未找到backend目录
    pause
    exit /b 1
)

if not exist "frontend" (
    echo [ERROR] 未找到frontend目录
    pause
    exit /b 1
)

if not exist "models" (
    echo [WARNING] 未找到models目录，某些功能可能受限
)

if not exist "training" (
    echo [WARNING] 未找到training目录，训练功能可能受限
)

echo [SUCCESS] 项目目录结构检查通过

echo.

REM ========================================================
REM 第二阶段: 虚拟环境和依赖管理
REM ========================================================
echo [阶段2] 虚拟环境和依赖管理
echo.

REM 检查虚拟环境
set VENV_DIR=.venv
set VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat

if exist "%VENV_DIR%" (
    echo [INFO] 虚拟环境已存在
    set /p RECREATE_VENV=是否重新创建虚拟环境？[y/N]: 
    if /i "%RECREATE_VENV%"=="y" (
        echo [INFO] 删除现有虚拟环境...
        rmdir /s /q "%VENV_DIR%"
        goto create_venv
    ) else (
        echo [INFO] 使用现有虚拟环境
        goto activate_venv
    )
) else (
    goto create_venv
)

:create_venv
echo [INFO] 创建Python虚拟环境...
%PYTHON_CMD% -m venv "%VENV_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] 虚拟环境创建失败
    pause
    exit /b 1
)
echo [SUCCESS] 虚拟环境创建成功

:activate_venv
echo [INFO] 激活虚拟环境...
if exist "%VENV_ACTIVATE%" (
    call "%VENV_ACTIVATE%"
    echo [SUCCESS] 虚拟环境已激活
) else (
    echo [ERROR] 虚拟环境激活文件不存在
    pause
    exit /b 1
)

echo.

REM 升级pip
echo [INFO] 升级pip...
python -m pip install --upgrade pip
if %errorlevel% equ 0 (
    echo [SUCCESS] pip升级完成
) else (
    echo [WARNING] pip升级失败，继续执行
)

echo.

REM 安装Python依赖
echo [INFO] 安装Python核心依赖...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if %errorlevel% equ 0 (
        echo [SUCCESS] Python核心依赖安装完成
    ) else (
        echo [ERROR] Python核心依赖安装失败
        pause
        exit /b 1
    )
) else (
    echo [WARNING] 未找到requirements.txt文件
)

if exist "pyproject.toml" (
    echo [INFO] 安装pyproject.toml中的依赖...
    python -m pip install -e .
    if %errorlevel% equ 0 (
        echo [SUCCESS] pyproject.toml依赖安装完成
    ) else (
        echo [WARNING] pyproject.toml依赖安装失败，继续执行
    )
)

echo.

REM 可选依赖安装
echo [INFO] 可选依赖安装...
set /p INSTALL_DEV=是否安装开发依赖？[y/N]: 
if /i "%INSTALL_DEV%"=="y" (
    echo [INFO] 安装开发依赖...
    python -m pip install -e ".[dev]"
    if %errorlevel% equ 0 (
        echo [SUCCESS] 开发依赖安装完成
    ) else (
        echo [WARNING] 开发依赖安装失败，继续执行
    )
)

if %GPU_ENABLED% equ 1 (
    set /p INSTALL_GPU=是否安装GPU支持依赖？[y/N]: 
    if /i "%INSTALL_GPU%"=="y" (
        echo [INFO] 安装GPU支持依赖...
        python -m pip install -e ".[gpu]"
        if %errorlevel% equ 0 (
            echo [SUCCESS] GPU支持依赖安装完成
        ) else (
            echo [WARNING] GPU支持依赖安装失败，继续执行
        )
    )
)

set /p INSTALL_TRAINING=是否安装训练相关依赖？[y/N]: 
if /i "%INSTALL_TRAINING%"=="y" (
    echo [INFO] 安装训练相关依赖...
    python -m pip install -e ".[training]"
    if %errorlevel% equ 0 (
        echo [SUCCESS] 训练相关依赖安装完成
    ) else (
        echo [WARNING] 训练相关依赖安装失败，继续执行
    )
)

echo.

REM 安装前端依赖
echo [INFO] 安装前端依赖...
if exist "frontend\package.json" (
    cd frontend
    echo [INFO] 当前目录: %cd%
    call npm install
    if %errorlevel% equ 0 (
        echo [SUCCESS] 前端依赖安装完成
    ) else (
        echo [ERROR] 前端依赖安装失败
        cd ..
        pause
        exit /b 1
    )
    cd ..
) else (
    echo [WARNING] 未找到前端package.json文件
)

echo.

REM ========================================================
REM 第三阶段: 数据库和系统初始化
REM ========================================================
echo [阶段3] 数据库和系统初始化
echo.

REM 检查环境配置文件
echo [INFO] 检查环境配置文件...
if not exist ".env" (
    echo [INFO] 创建环境配置文件...
    (
        echo # Self AGI 环境配置
        echo # ====================
        echo # 安全警告：生产环境必须修改所有密码和密钥！
        echo # ====================
        echo.
        echo # 数据库配置（开发环境使用SQLite）
        echo DATABASE_URL=sqlite:///./self_agi.db
        echo.
        echo # 后端配置
        echo SECRET_KEY=your_secret_key_change_in_production
        echo ENVIRONMENT=development
        echo ALLOWED_HOSTS=*
        echo CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:8000
        echo.
        echo # 前端配置
        echo VITE_API_URL=http://localhost:8000/api
        echo VITE_TRAINING_API_URL=http://localhost:8001
        echo.
        echo # JWT配置
        echo ACCESS_TOKEN_EXPIRE_MINUTES=1440
        echo REFRESH_TOKEN_EXPIRE_DAYS=7
        echo.
        echo # 训练配置
        echo TRAINING_GPU_ENABLED=%GPU_ENABLED%
        echo TRAINING_MODEL_DIR=./models
        echo.
        echo # 硬件配置
        echo HARDWARE_SERIAL_PORT=COM3
        echo HARDWARE_BAUDRATE=115200
        echo.
        echo # 用户账户配置（可选，用于create_admin.py脚本）
        echo # 管理员账户
        echo # ADMIN_PASSWORD=your_secure_admin_password
        echo # ADMIN_EMAIL=admin@yourdomain.com
        echo # 演示用户账户
        echo # DEMO_PASSWORD=your_secure_demo_password
        echo # DEMO_EMAIL=demo@yourdomain.com
        echo # DEMO_IS_ADMIN=false
    ) > .env
    echo [SUCCESS] 环境配置文件 .env 已创建
) else (
    echo [INFO] 环境配置文件 .env 已存在
)

echo.

REM 初始化数据库
echo [INFO] 初始化数据库...
if exist "backend\create_missing_tables.py" (
    python backend/create_missing_tables.py
    if %errorlevel% equ 0 (
        echo [SUCCESS] 数据库表创建完成
    ) else (
        echo [WARNING] 数据库表创建失败，继续执行
    )
) else (
    echo [WARNING] 未找到数据库初始化脚本
)

if exist "backend\create_admin.py" (
    echo [INFO] 创建管理员账户...
    python backend/create_admin.py
    if %errorlevel% equ 0 (
        echo [SUCCESS] 管理员账户创建完成
    ) else (
        echo [WARNING] 管理员账户创建失败，继续执行
    )
)

echo.

REM ========================================================
REM 第四阶段: 服务启动和系统监控
REM ========================================================
echo [阶段4] 服务启动和系统监控
echo.

REM 选择启动模式
echo [INFO] 选择启动模式:
echo   1. 开发模式 (后端: uvicorn, 前端: vite dev)
echo   2. 生产模式 (后端: gunicorn, 前端: vite build + serve)
echo   3. 仅启动后端服务
echo   4. 仅启动前端服务
echo   5. 启动所有服务 (开发模式)
echo.
set /p START_MODE=请输入选择 [1-5] (默认: 5): 
if "%START_MODE%"=="" set START_MODE=5

REM 创建日志目录
if not exist "logs" mkdir logs

echo.

REM 根据选择启动服务
if "%START_MODE%"=="1" goto start_dev_mode
if "%START_MODE%"=="2" goto start_prod_mode
if "%START_MODE%"=="3" goto start_backend_only
if "%START_MODE%"=="4" goto start_frontend_only
if "%START_MODE%"=="5" goto start_all_dev
goto start_all_dev

:start_dev_mode
echo [INFO] 启动开发模式...
echo [INFO] 启动后端开发服务器 (uvicorn)...
start "Self AGI 后端 (开发)" cmd /k "cd backend & uvicorn main:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul
echo [INFO] 启动前端开发服务器...
start "Self AGI 前端 (开发)" cmd /k "cd frontend && npm run dev"
goto services_info

:start_prod_mode
echo [INFO] 启动生产模式...
echo [INFO] 构建前端...
cd frontend
call npm run build
if %errorlevel% neq 0 (
    echo [ERROR] 前端构建失败
    cd ..
    pause
    exit /b 1
)
cd ..
echo [SUCCESS] 前端构建完成

echo [INFO] 启动后端生产服务器 (gunicorn)...
if exist "backend\gunicorn_config.py" (
    start "Self AGI 后端 (生产)" cmd /k "gunicorn -c backend/gunicorn_config.py backend.main:app"
) else (
    echo [WARNING] 未找到gunicorn配置，使用默认配置
    start "Self AGI 后端 (生产)" cmd /k "gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 backend.main:app"
)

echo [INFO] 启动前端生产服务器 (serve)...
if exist "frontend\node_modules\.bin\serve.cmd" (
    start "Self AGI 前端 (生产)" cmd /k "cd frontend && npx serve -s dist -l 3000"
) else (
    echo [INFO] 安装serve工具...
    cd frontend
    npm install -g serve
    cd ..
    start "Self AGI 前端 (生产)" cmd /k "cd frontend && serve -s dist -l 3000"
)
goto services_info

:start_backend_only
echo [INFO] 仅启动后端服务...
echo [INFO] 启动后端开发服务器...
start "Self AGI 后端" cmd /k "cd backend & uvicorn main:app --host 0.0.0.0 --port 8000"
goto services_info

:start_frontend_only
echo [INFO] 仅启动前端服务...
echo [INFO] 启动前端开发服务器...
start "Self AGI 前端" cmd /k "cd frontend && npm run dev"
goto services_info

:start_all_dev
echo [INFO] 启动所有服务 (开发模式)...
echo [INFO] 启动后端开发服务器...
start "Self AGI 后端" cmd /k "cd backend & uvicorn main:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul
echo [INFO] 启动前端开发服务器...
start "Self AGI 前端" cmd /k "cd frontend && npm run dev"
timeout /t 2 /nobreak >nul
echo [INFO] 启动训练服务 (可选)...
if exist "training\training_server.py" (
    set /p START_TRAINING=是否启动训练服务？[y/N]: 
    if /i "%START_TRAINING%"=="y" (
        start "Self AGI 训练服务" cmd /k "python training/training_server.py"
    )
)

:services_info
echo.

REM ========================================================
REM 第五阶段: 系统信息和服务状态
REM ========================================================
echo [阶段5] 系统信息和服务状态
echo.

echo ========================================================
echo                Self AGI 系统启动完成
echo ========================================================
echo.
echo [系统信息]
echo   启动时间: %date% %time%
echo   Python版本: %PYTHON_VERSION%
echo   Node.js版本: %NODE_VERSION%
echo   GPU支持: %GPU_ENABLED% (0=禁用, 1=启用)
echo   虚拟环境: %VENV_DIR%
echo.
echo [服务信息]
if "%START_MODE%"=="1" (
    echo   后端: http://localhost:8000 (开发模式)
    echo   前端: http://localhost:3000 (开发模式)
    echo   API文档: http://localhost:8000/docs
) else if "%START_MODE%"=="2" (
    echo   后端: http://localhost:8000 (生产模式)
    echo   前端: http://localhost:3000 (生产模式)
    echo   API文档: http://localhost:8000/docs
) else if "%START_MODE%"=="3" (
    echo   后端: http://localhost:8000
    echo   API文档: http://localhost:8000/docs
) else if "%START_MODE%"=="4" (
    echo   前端: http://localhost:3000
) else (
    echo   后端: http://localhost:8000
    echo   前端: http://localhost:3000
    echo   API文档: http://localhost:8000/docs
)
echo.
echo [默认登录信息]
echo   管理员用户名: admin
echo   管理员邮箱: admin@selfagi.com (可通过ADMIN_EMAIL环境变量修改)
echo   演示用户名: demo
echo   演示用户邮箱: demo@selfagi.com (可通过DEMO_EMAIL环境变量修改)
echo   密码安全性: 默认密码已隐藏，建议通过环境变量设置安全密码
echo.
echo [重要说明]
echo   1. 生产环境必须通过环境变量设置安全密码（ADMIN_PASSWORD/DEMO_PASSWORD）
echo   2. 必须修改.env文件中的SECRET_KEY和其他安全配置
echo   3. 查看日志文件: logs\ 目录
echo   4. 停止服务: 关闭对应命令行窗口
echo   5. 演示用户默认没有管理员权限，可通过DEMO_IS_ADMIN=true启用
echo.
echo ========================================================
echo.

REM 打开浏览器
set /p OPEN_BROWSER=是否打开浏览器访问系统？[Y/n]: 
if /i "%OPEN_BROWSER%"=="n" (
    echo [INFO] 跳过浏览器打开
) else (
    if "%START_MODE%"=="3" (
        start http://localhost:8000/docs
    ) else if "%START_MODE%"=="4" (
        start http://localhost:3000
    ) else (
        start http://localhost:3000
    )
)

echo.
echo [INFO] 按任意键退出本脚本，服务将继续在后台运行...
pause >nul

exit /b 0

REM ========================================================
REM 错误处理
REM ========================================================
:error_exit
echo [ERROR] 脚本执行失败
echo [INFO] 错误代码: %errorlevel%
echo [INFO] 请检查日志和错误信息
pause
exit /b 1