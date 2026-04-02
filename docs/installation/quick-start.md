# Quick Start | 快速开始

This guide provides quick installation and setup instructions for the Self AGI system.

本指南提供 Self AGI 系统的快速安装和设置说明。

## Prerequisites | 先决条件

### Hardware Requirements | 硬件要求
- **CPU**: x86-64 processor with SSE4.2 support
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: Minimum 20GB free space
- **GPU**: Optional, but recommended for training (NVIDIA with CUDA support)

- **CPU**: 支持 SSE4.2 的 x86-64 处理器
- **内存**: 最低 8GB，推荐 16GB+
- **存储**: 最低 20GB 可用空间
- **GPU**: 可选，但推荐用于训练（支持CUDA的NVIDIA显卡）

### Software Requirements | 软件要求
- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04, or macOS 12+
- **Python**: 3.9 or higher
- **Node.js**: 18 or higher (for frontend)
- **Git**: Latest version

- **操作系统**: Windows 10/11、Ubuntu 20.04/22.04 或 macOS 12+
- **Python**: 3.9 或更高版本
- **Node.js**: 18 或更高版本（用于前端）
- **Git**: 最新版本

## Installation Steps | 安装步骤

### 1. Clone Repository | 克隆仓库
```bash
git clone https://github.com/Sum-Outman/Self.git
cd Self
```

### 2. Set Up Python Environment | 设置Python环境
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Install Python Dependencies | 安装Python依赖
```bash
# Install core dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"

# Install GPU dependencies (optional, if you have CUDA)
pip install -e ".[gpu]"
```

### 4. Install Frontend Dependencies | 安装前端依赖
```bash
cd frontend
npm install
cd ..
```

### 5. Initialize Database | 初始化数据库
```bash
python backend/create_missing_tables.py
python backend/create_admin.py
```

### 6. Start Backend Server | 启动后端服务器
```bash
python backend/main.py
```
The backend will start at http://localhost:8000

后端将在 http://localhost:8000 启动

### 7. Start Frontend Server | 启动前端服务器
```bash
cd frontend
npm run dev
```
The frontend will start at http://localhost:3000

前端将在 http://localhost:3000 启动

## Verify Installation | 验证安装

### Check Backend Status | 检查后端状态
Open browser and visit http://localhost:8000/docs
You should see the Swagger UI API documentation.

打开浏览器访问 http://localhost:8000/docs
您应该看到 Swagger UI API 文档。

### Check Frontend Status | 检查前端状态
Open browser and visit http://localhost:3000
You should see the Self AGI login page.

打开浏览器访问 http://localhost:3000
您应该看到 Self AGI 登录页面。

### Login | 登录
- **Email**: admin@example.com
- **Password**: admin123

- **邮箱**: admin@example.com
- **密码**: admin123

## Troubleshooting | 故障排除

### Common Issues | 常见问题

#### Port Already in Use | 端口已被占用
If port 8000 or 3000 is already in use:

如果端口 8000 或 3000 已被占用：

```bash
# Change backend port
python backend/main.py --port 8001

# Change frontend port
cd frontend
npm run dev -- --port 3001
```

#### Database Connection Error | 数据库连接错误
If you see database connection errors:

如果看到数据库连接错误：

```bash
# Recreate database
rm self_agi.db  # Delete existing database
python backend/create_missing_tables.py
```

#### Module Import Error | 模块导入错误
If you see import errors:

如果看到导入错误：

```bash
# Reinstall dependencies
pip uninstall -y self-agi
pip install -e .
```

## Next Steps | 后续步骤

After successful installation:

安装成功后：

1. **Explore Features**: Try out the AGI chat, training, and hardware control features
2. **Read Documentation**: Read the full documentation for detailed usage instructions
3. **Run Examples**: Run example scripts in the `examples/` directory
4. **Contribute**: Check the contribution guidelines if you want to contribute

1. **探索功能**: 尝试 AGI 聊天、训练和硬件控制功能
2. **阅读文档**: 阅读完整文档获取详细使用说明
3. **运行示例**: 运行 `examples/` 目录中的示例脚本
4. **贡献**: 如果您想贡献，请查看贡献指南

## Getting Help | 获取帮助

If you encounter issues:

如果遇到问题：

1. **Check Issues**: Check existing issues on GitHub
2. **Create Issue**: Create a new issue with detailed information
3. **Contact**: Email silencecrowtom@qq.com for support

1. **检查问题**: 检查 GitHub 上的现有问题
2. **创建问题**: 创建包含详细信息的新问题
3. **联系**: 发送邮件至 silencecrowtom@qq.com 获取支持

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*