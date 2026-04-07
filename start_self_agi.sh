#!/bin/bash
# Self AGI 完整部署启动脚本 - 版本 1.0
# 支持 Linux 和 macOS 系统

set -e  # 遇到错误时退出脚本

# ========================================================
# 颜色定义
# ========================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ========================================================
# 系统信息
# ========================================================
clear
echo "========================================================"
echo "          Self AGI 完整部署启动脚本"
echo "========================================================"
echo "系统概述: 完整的自主通用人工智能系统"
echo "版本: v1.3 (安全与功能全面增强版)"
echo "最后更新: 2026年3月29日"
echo "许可证: Apache License 2.0"
echo "邮箱: silenceceowtom@qq.com"
echo "支持系统: Linux, macOS"
echo "========================================================"
echo ""
echo "硬件处理逻辑:"
echo "  • 系统可在无硬件条件下运行AGI功能"
echo "  • 硬件部分无任何虚拟数据，完全禁用降级处理"
echo "  • 连接硬件后自动切换至真实硬件接口"
echo "  • 支持部分硬件连接工作模式"
echo "  • 一次训练兼容多型号人形机器人"
echo ""

# ========================================================
# 第一阶段: 系统环境检查和依赖验证
# ========================================================
print_info "阶段1: 系统环境检查和依赖验证"
echo ""

# 检查操作系统类型
print_info "检测操作系统..."
OS_TYPE="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
    print_success "检测到 Linux 系统"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
    print_success "检测到 macOS 系统"
else
    print_warning "未知操作系统类型: $OSTYPE"
    OS_TYPE="other"
fi

# 检查Python版本
print_info "检查Python版本..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python版本: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_success "Python版本: $PYTHON_VERSION"
    PYTHON_CMD="python"
else
    print_error "未找到Python，请安装Python 3.9+"
    echo "下载链接: https://www.python.org/downloads/"
    exit 1
fi

# 验证Python版本号
print_info "验证Python版本要求（需要3.9+）..."
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ $PYTHON_MAJOR -lt 3 ]; then
    print_error "Python版本不符合要求，需要Python 3.9+"
    exit 1
elif [ $PYTHON_MAJOR -eq 3 ] && [ $PYTHON_MINOR -lt 9 ]; then
    print_error "Python版本不符合要求，需要Python 3.9+"
    exit 1
fi
print_success "Python版本检查通过"

echo ""

# 检查Node.js版本
print_info "检查Node.js版本..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js版本: $NODE_VERSION"
else
    print_error "未找到Node.js，请安装Node.js 18+"
    echo "下载链接: https://nodejs.org/"
    exit 1
fi

# 验证Node.js版本
print_info "验证Node.js版本要求（需要18+）..."
NODE_MAJOR=$(node -e "const v = process.version.match(/v(\d+)\./)[1]; console.log(v);")
if [ $NODE_MAJOR -lt 18 ]; then
    print_error "Node.js版本不符合要求，需要Node.js 18+"
    exit 1
fi
print_success "Node.js版本检查通过"

echo ""

# 检查npm版本
print_info "检查npm版本..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_success "npm版本: $NPM_VERSION"
else
    print_error "未找到npm，请确保Node.js安装正确"
    exit 1
fi

echo ""

# 检查CUDA（可选）
print_info "检查CUDA支持（GPU训练和推理）..."
if command -v nvidia-smi &> /dev/null; then
    print_info "检测到NVIDIA GPU，检查CUDA版本..."
    CUDA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    print_success "NVIDIA驱动版本: $CUDA_DRIVER_VERSION"
    
    CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        print_success "PyTorch CUDA支持已启用"
        GPU_ENABLED=1
    else
        print_warning "PyTorch CUDA支持未启用，将使用CPU模式"
        GPU_ENABLED=0
    fi
else
    print_info "未检测到NVIDIA GPU或nvidia-smi不可用，将使用CPU模式"
    GPU_ENABLED=0
fi

echo ""

# 检查系统资源
print_info "检查系统资源..."
if [ "$OS_TYPE" = "linux" ]; then
    MEM_INFO=$(free -h | awk '/^Mem:/ {print "可用内存: " $7}')
    CPU_INFO=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
elif [ "$OS_TYPE" = "macos" ]; then
    MEM_INFO=$(sysctl -n hw.memsize | awk '{printf "总内存: %.2f GB", $1/1024/1024/1024}')
    CPU_INFO=$(sysctl -n machdep.cpu.brand_string)
else
    MEM_INFO="未知"
    CPU_INFO="未知"
fi
print_info "系统内存: $MEM_INFO"
print_info "CPU信息: $CPU_INFO"

echo ""

# 检查必要目录
print_info "检查项目目录结构..."
if [ ! -d "backend" ]; then
    print_error "未找到backend目录"
    exit 1
fi

if [ ! -d "frontend" ]; then
    print_error "未找到frontend目录"
    exit 1
fi

if [ ! -d "models" ]; then
    print_warning "未找到models目录，某些功能可能受限"
fi

if [ ! -d "training" ]; then
    print_warning "未找到training目录，训练功能可能受限"
fi

print_success "项目目录结构检查通过"

echo ""

# ========================================================
# 第二阶段: 虚拟环境和依赖管理
# ========================================================
print_info "阶段2: 虚拟环境和依赖管理"
echo ""

# 检查虚拟环境
VENV_DIR=".venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"

if [ -d "$VENV_DIR" ]; then
    print_info "虚拟环境已存在"
    read -p "是否重新创建虚拟环境？[y/N]: " RECREATE_VENV
    if [[ "$RECREATE_VENV" =~ ^[Yy]$ ]]; then
        print_info "删除现有虚拟环境..."
        rm -rf "$VENV_DIR"
    else
        print_info "使用现有虚拟环境"
    fi
fi

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    print_info "创建Python虚拟环境..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        print_error "虚拟环境创建失败"
        exit 1
    fi
    print_success "虚拟环境创建成功"
fi

# 激活虚拟环境
print_info "激活虚拟环境..."
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
    print_success "虚拟环境已激活"
else
    print_error "虚拟环境激活文件不存在"
    exit 1
fi

echo ""

# 升级pip
print_info "升级pip..."
python -m pip install --upgrade pip
if [ $? -eq 0 ]; then
    print_success "pip升级完成"
else
    print_warning "pip升级失败，继续执行"
fi

echo ""

# 安装Python依赖
print_info "安装Python核心依赖..."
if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_success "Python核心依赖安装完成"
    else
        print_error "Python核心依赖安装失败"
        exit 1
    fi
else
    print_warning "未找到requirements.txt文件"
fi

if [ -f "pyproject.toml" ]; then
    print_info "安装pyproject.toml中的依赖..."
    python -m pip install -e .
    if [ $? -eq 0 ]; then
        print_success "pyproject.toml依赖安装完成"
    else
        print_warning "pyproject.toml依赖安装失败，继续执行"
    fi
fi

echo ""

# 可选依赖安装
print_info "可选依赖安装..."
read -p "是否安装开发依赖？[y/N]: " INSTALL_DEV
if [[ "$INSTALL_DEV" =~ ^[Yy]$ ]]; then
    print_info "安装开发依赖..."
    python -m pip install -e ".[dev]"
    if [ $? -eq 0 ]; then
        print_success "开发依赖安装完成"
    else
        print_warning "开发依赖安装失败，继续执行"
    fi
fi

if [ $GPU_ENABLED -eq 1 ]; then
    read -p "是否安装GPU支持依赖？[y/N]: " INSTALL_GPU
    if [[ "$INSTALL_GPU" =~ ^[Yy]$ ]]; then
        print_info "安装GPU支持依赖..."
        python -m pip install -e ".[gpu]"
        if [ $? -eq 0 ]; then
            print_success "GPU支持依赖安装完成"
        else
            print_warning "GPU支持依赖安装失败，继续执行"
        fi
    fi
fi

read -p "是否安装训练相关依赖？[y/N]: " INSTALL_TRAINING
if [[ "$INSTALL_TRAINING" =~ ^[Yy]$ ]]; then
    print_info "安装训练相关依赖..."
    python -m pip install -e ".[training]"
    if [ $? -eq 0 ]; then
        print_success "训练相关依赖安装完成"
    else
        print_warning "训练相关依赖安装失败，继续执行"
    fi
fi

echo ""

# 安装前端依赖
print_info "安装前端依赖..."
if [ -f "frontend/package.json" ]; then
    cd frontend
    print_info "当前目录: $(pwd)"
    npm install
    if [ $? -eq 0 ]; then
        print_success "前端依赖安装完成"
    else
        print_error "前端依赖安装失败"
        cd ..
        exit 1
    fi
    cd ..
else
    print_warning "未找到前端package.json文件"
fi

echo ""

# ========================================================
# 第三阶段: 数据库和系统初始化
# ========================================================
print_info "阶段3: 数据库和系统初始化"
echo ""

# 检查环境配置文件
print_info "检查环境配置文件..."
if [ ! -f ".env" ]; then
    print_info "创建环境配置文件..."
    cat > .env << EOF
# Self AGI 环境配置
# ====================
# 安全警告：生产环境必须修改所有密码和密钥！
# ====================

# 数据库配置（开发环境使用SQLite）
DATABASE_URL=sqlite:///./self_agi.db

# 后端配置
SECRET_KEY=your_secret_key_change_in_production
ENVIRONMENT=development
ALLOWED_HOSTS=*
CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:8000

# 前端配置
VITE_API_URL=http://localhost:8000/api
VITE_TRAINING_API_URL=http://localhost:8001

# JWT配置
ACCESS_TOKEN_EXPIRE_MINUTES=1440
REFRESH_TOKEN_EXPIRE_DAYS=7

# 训练配置
TRAINING_GPU_ENABLED=$GPU_ENABLED
TRAINING_MODEL_DIR=./models

# 硬件配置
HARDWARE_SERIAL_PORT=/dev/ttyUSB0
HARDWARE_BAUDRATE=115200
EOF
    print_success "环境配置文件 .env 已创建"
else
    print_info "环境配置文件 .env 已存在"
fi

echo ""

# 初始化数据库
print_info "初始化数据库..."
if [ -f "backend/create_missing_tables.py" ]; then
    python backend/create_missing_tables.py
    if [ $? -eq 0 ]; then
        print_success "数据库表创建完成"
    else
        print_warning "数据库表创建失败，继续执行"
    fi
else
    print_warning "未找到数据库初始化脚本"
fi

if [ -f "backend/create_admin.py" ]; then
    print_info "创建管理员账户..."
    python backend/create_admin.py
    if [ $? -eq 0 ]; then
        print_success "管理员账户创建完成"
    else
        print_warning "管理员账户创建失败，继续执行"
    fi
fi

echo ""

# ========================================================
# 第四阶段: 服务启动和系统监控
# ========================================================
print_info "阶段4: 服务启动和系统监控"
echo ""

# 选择启动模式
echo "选择启动模式:"
echo "  1. 开发模式 (后端: uvicorn, 前端: vite dev)"
echo "  2. 生产模式 (后端: gunicorn, 前端: vite build + serve)"
echo "  3. 仅启动后端服务"
echo "  4. 仅启动前端服务"
echo "  5. 启动所有服务 (开发模式)"
echo ""
read -p "请输入选择 [1-5] (默认: 5): " START_MODE
START_MODE=${START_MODE:-5}

# 创建日志目录
mkdir -p logs

echo ""

# 根据选择启动服务
case $START_MODE in
    1)
        print_info "启动开发模式..."
        print_info "启动后端开发服务器 (uvicorn)..."
        gnome-terminal --tab --title="Self AGI 后端 (开发)" -- bash -c "python backend/main.py; exec bash" 2>/dev/null || \
        xterm -title "Self AGI 后端 (开发)" -e "python backend/main.py; exec bash" 2>/dev/null || \
        screen -dmS self_agi_backend python backend/main.py
        sleep 3
        print_info "启动前端开发服务器..."
        gnome-terminal --tab --title="Self AGI 前端 (开发)" -- bash -c "cd frontend && npm run dev; exec bash" 2>/dev/null || \
        xterm -title "Self AGI 前端 (开发)" -e "cd frontend && npm run dev; exec bash" 2>/dev/null || \
        screen -dmS self_agi_frontend bash -c "cd frontend && npm run dev"
        ;;
    2)
        print_info "启动生产模式..."
        print_info "构建前端..."
        cd frontend
        npm run build
        if [ $? -ne 0 ]; then
            print_error "前端构建失败"
            cd ..
            exit 1
        fi
        cd ..
        print_success "前端构建完成"

        print_info "启动后端生产服务器 (gunicorn)..."
        if [ -f "backend/gunicorn_config.py" ]; then
            gnome-terminal --tab --title="Self AGI 后端 (生产)" -- bash -c "gunicorn -c backend/gunicorn_config.py backend.main:app; exec bash" 2>/dev/null || \
            xterm -title "Self AGI 后端 (生产)" -e "gunicorn -c backend/gunicorn_config.py backend.main:app; exec bash" 2>/dev/null || \
            screen -dmS self_agi_backend_prod gunicorn -c backend/gunicorn_config.py backend.main:app
        else
            print_warning "未找到gunicorn配置，使用默认配置"
            gnome-terminal --tab --title="Self AGI 后端 (生产)" -- bash -c "gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 backend.main:app; exec bash" 2>/dev/null || \
            xterm -title "Self AGI 后端 (生产)" -e "gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 backend.main:app; exec bash" 2>/dev/null || \
            screen -dmS self_agi_backend_prod gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 backend.main:app
        fi

        print_info "启动前端生产服务器 (serve)..."
        if [ -f "frontend/node_modules/.bin/serve" ]; then
            gnome-terminal --tab --title="Self AGI 前端 (生产)" -- bash -c "cd frontend && npx serve -s dist -l 3000; exec bash" 2>/dev/null || \
            xterm -title "Self AGI 前端 (生产)" -e "cd frontend && npx serve -s dist -l 3000; exec bash" 2>/dev/null || \
            screen -dmS self_agi_frontend_prod bash -c "cd frontend && npx serve -s dist -l 3000"
        else
            print_info "安装serve工具..."
            cd frontend
            npm install -g serve
            cd ..
            gnome-terminal --tab --title="Self AGI 前端 (生产)" -- bash -c "cd frontend && serve -s dist -l 3000; exec bash" 2>/dev/null || \
            xterm -title "Self AGI 前端 (生产)" -e "cd frontend && serve -s dist -l 3000; exec bash" 2>/dev/null || \
            screen -dmS self_agi_frontend_prod bash -c "cd frontend && serve -s dist -l 3000"
        fi
        ;;
    3)
        print_info "仅启动后端服务..."
        print_info "启动后端开发服务器..."
        gnome-terminal --tab --title="Self AGI 后端" -- bash -c "python backend/main.py; exec bash" 2>/dev/null || \
        xterm -title "Self AGI 后端" -e "python backend/main.py; exec bash" 2>/dev/null || \
        screen -dmS self_agi_backend python backend/main.py
        ;;
    4)
        print_info "仅启动前端服务..."
        print_info "启动前端开发服务器..."
        gnome-terminal --tab --title="Self AGI 前端" -- bash -c "cd frontend && npm run dev; exec bash" 2>/dev/null || \
        xterm -title "Self AGI 前端" -e "cd frontend && npm run dev; exec bash" 2>/dev/null || \
        screen -dmS self_agi_frontend bash -c "cd frontend && npm run dev"
        ;;
    5|*)
        print_info "启动所有服务 (开发模式)..."
        print_info "启动后端开发服务器..."
        gnome-terminal --tab --title="Self AGI 后端" -- bash -c "python backend/main.py; exec bash" 2>/dev/null || \
        xterm -title "Self AGI 后端" -e "python backend/main.py; exec bash" 2>/dev/null || \
        screen -dmS self_agi_backend python backend/main.py
        sleep 3
        print_info "启动前端开发服务器..."
        gnome-terminal --tab --title="Self AGI 前端" -- bash -c "cd frontend && npm run dev; exec bash" 2>/dev/null || \
        xterm -title "Self AGI 前端" -e "cd frontend && npm run dev; exec bash" 2>/dev/null || \
        screen -dmS self_agi_frontend bash -c "cd frontend && npm run dev"
        sleep 2
        print_info "启动训练服务 (可选)..."
        if [ -f "training/training_server.py" ]; then
            read -p "是否启动训练服务？[y/N]: " START_TRAINING
            if [[ "$START_TRAINING" =~ ^[Yy]$ ]]; then
                gnome-terminal --tab --title="Self AGI 训练服务" -- bash -c "python training/training_server.py; exec bash" 2>/dev/null || \
                xterm -title "Self AGI 训练服务" -e "python training/training_server.py; exec bash" 2>/dev/null || \
                screen -dmS self_agi_training python training/training_server.py
            fi
        fi
        ;;
esac

echo ""

# ========================================================
# 第五阶段: 系统信息和服务状态
# ========================================================
print_info "阶段5: 系统信息和服务状态"
echo ""

echo "========================================================"
echo "                Self AGI 系统启动完成"
echo "========================================================"
echo ""
echo "系统信息"
echo "  启动时间: $(date)"
echo "  Python版本: $PYTHON_VERSION"
echo "  Node.js版本: $NODE_VERSION"
echo "  GPU支持: $GPU_ENABLED (0=禁用, 1=启用)"
echo "  操作系统: $OS_TYPE"
echo "  虚拟环境: $VENV_DIR"
echo ""
echo "服务信息"
case $START_MODE in
    1)
        echo "  后端: http://localhost:8000 (开发模式)"
        echo "  前端: http://localhost:3000 (开发模式)"
        echo "  API文档: http://localhost:8000/docs"
        ;;
    2)
        echo "  后端: http://localhost:8000 (生产模式)"
        echo "  前端: http://localhost:3000 (生产模式)"
        echo "  API文档: http://localhost:8000/docs"
        ;;
    3)
        echo "  后端: http://localhost:8000"
        echo "  API文档: http://localhost:8000/docs"
        ;;
    4)
        echo "  前端: http://localhost:3000"
        ;;
    5|*)
        echo "  后端: http://localhost:8000"
        echo "  前端: http://localhost:3000"
        echo "  API文档: http://localhost:8000/docs"
        ;;
esac
echo ""
echo "默认登录信息"
echo "  管理员邮箱: admin@example.com"
echo "  管理员密码: admin123"
echo ""
echo "重要说明"
echo "  1. 首次启动后请立即修改默认密码"
echo "  2. 生产环境必须修改.env文件中的安全配置"
echo "  3. 查看日志文件: logs/ 目录"
echo "  4. 停止服务: 使用 pkill 命令或关闭终端"
echo ""
echo "========================================================"
echo ""

# 打开浏览器
read -p "是否打开浏览器访问系统？[Y/n]: " OPEN_BROWSER
OPEN_BROWSER=${OPEN_BROWSER:-Y}
if [[ "$OPEN_BROWSER" =~ ^[Nn]$ ]]; then
    print_info "跳过浏览器打开"
else
    case $START_MODE in
        3)
            if command -v xdg-open &> /dev/null; then
                xdg-open "http://localhost:8000/docs"
            elif command -v open &> /dev/null; then
                open "http://localhost:8000/docs"
            fi
            ;;
        4)
            if command -v xdg-open &> /dev/null; then
                xdg-open "http://localhost:3000"
            elif command -v open &> /dev/null; then
                open "http://localhost:3000"
            fi
            ;;
        *)
            if command -v xdg-open &> /dev/null; then
                xdg-open "http://localhost:3000"
            elif command -v open &> /dev/null; then
                open "http://localhost:3000"
            fi
            ;;
    esac
fi

echo ""
print_info "脚本执行完成，服务已在后台启动"
print_info "按任意键退出本脚本..."
read -n 1 -s

exit 0