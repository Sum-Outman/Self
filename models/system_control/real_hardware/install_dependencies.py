#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件依赖安装脚本

一键安装真实硬件接口所需依赖
支持跨平台安装
"""

import subprocess
import sys
import platform
import os

def get_platform_info():
    """获取平台信息"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # 检查是否是树莓派
    is_raspberry_pi = False
    if system == "linux":
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read()
                if "raspberry pi" in model.lower():
                    is_raspberry_pi = True
        except:
            pass  # 已实现
    
    return {
        "system": system,
        "machine": machine,
        "is_raspberry_pi": is_raspberry_pi,
        "python_version": platform.python_version(),
    }


def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def install_pip_package(package_name):
    """安装pip包"""
    print(f"安装 {package_name}...")
    
    # 检查是否已安装
    check_cmd = f"{sys.executable} -c \"import {package_name.replace('-', '_')}\" 2>&1"
    already_installed, stdout, stderr = run_command(check_cmd)
    
    if already_installed:
        print(f"  ✓ {package_name} 已安装")
        return True
    
    # 安装包
    install_cmd = f"{sys.executable} -m pip install {package_name} --quiet"
    success, stdout, stderr = run_command(install_cmd)
    
    if success:
        print(f"  ✓ {package_name} 安装成功")
        return True
    else:
        # 尝试升级pip后重试
        print(f"  ⚠ {package_name} 安装失败，尝试升级pip...")
        run_command(f"{sys.executable} -m pip install --upgrade pip --quiet")
        
        success, stdout, stderr = run_command(install_cmd)
        if success:
            print(f"  ✓ {package_name} 安装成功")
            return True
        else:
            print(f"  ✗ {package_name} 安装失败: {stderr[:100]}...")
            return False


def install_system_packages(platform_info):
    """安装系统包（Linux/macOS）"""
    system = platform_info["system"]
    
    if system == "linux":
        if platform_info["is_raspberry_pi"]:
            print("\n安装树莓派系统包...")
            
            packages = [
                "python3-dev", 
                "python3-pip",
                "python3-setuptools",
                "i2c-tools",
                "spi-tools",
                "libusb-1.0-0-dev",
                "libudev-dev",
            ]
            
            for pkg in packages:
                print(f"安装 {pkg}...")
                success, stdout, stderr = run_command(f"sudo apt-get install -y {pkg}")
                if success:
                    print(f"  ✓ {pkg}")
                else:
                    print(f"  ✗ {pkg}: {stderr[:100]}...")
        
        else:
            print("\n安装Linux系统包...")
            
            # Ubuntu/Debian
            packages = [
                "python3-dev",
                "python3-pip",
                "libusb-1.0-0-dev",
                "libudev-dev",
            ]
            
            for pkg in packages:
                print(f"安装 {pkg}...")
                success, stdout, stderr = run_command(f"sudo apt-get install -y {pkg} 2>/dev/null || echo '跳过 {pkg}'")
                print(f"  {'✓' if success else '⚠'} {pkg}")
    
    elif system == "darwin":  # macOS
        print("\n安装macOS系统包...")
        
        # 检查是否已安装Homebrew
        brew_installed, _, _ = run_command("which brew")
        if not brew_installed:
            print("  警告: Homebrew未安装，跳过系统包安装")
            print("  建议安装Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return
        
        packages = [
            "libusb",
            "pkg-config",
        ]
        
        for pkg in packages:
            print(f"安装 {pkg}...")
            success, stdout, stderr = run_command(f"brew install {pkg}")
            print(f"  {'✓' if success else '⚠'} {pkg}")
    
    else:
        print(f"\n系统 {system} 跳过系统包安装")


def install_windows_dependencies():
    """安装Windows特定依赖"""
    print("\n安装Windows特定依赖...")
    
    # Windows不需要特殊系统包
    print("  Windows依赖主要通过pip安装")
    
    # 建议安装Visual C++ Build Tools
    print("\nWindows开发环境建议:")
    print("  1. 安装Visual Studio Build Tools (C++编译环境)")
    print("  2. 或安装MinGW-w64")
    print("  3. 某些库可能需要编译，建议安装构建工具")


def install_hardware_dependencies(platform_info):
    """安装硬件依赖"""
    print("\n" + "="*60)
    print("安装硬件依赖")
    print("="*60)
    
    # 基础依赖
    print("\n1. 基础通信库:")
    base_deps = [
        "pyserial",      # 串口通信
        "pyusb",         # USB设备
        "python-can",    # CAN总线
    ]
    
    # 根据平台添加依赖
    if platform_info["system"] == "linux":
        if platform_info["is_raspberry_pi"]:
            base_deps.extend([
                "RPi.GPIO",     # 树莓派GPIO
                "smbus2",       # I2C通信
                "spidev",       # SPI通信
            ])
        else:
            # 非树莓派Linux可能不支持某些硬件
            print("  非树莓派Linux，跳过RPi.GPIO")
            base_deps.extend([
                "smbus2",
                "spidev",
            ])
    
    # 跨平台GPIO库
    base_deps.append("gpiozero")
    
    # 安装基础依赖
    for dep in base_deps:
        install_pip_package(dep)
    
    # 电机控制依赖
    print("\n2. 电机控制库:")
    motor_deps = [
        "Adafruit-PCA9685",  # PWM扩展板
    ]
    
    for dep in motor_deps:
        install_pip_package(dep)
    
    # 传感器依赖
    print("\n3. 传感器库:")
    sensor_deps = [
        "adafruit-circuitpython-ads1x15",  # ADC转换器
        "adafruit-circuitpython-dht",      # 温湿度传感器
        "opencv-python",                   # 摄像头
    ]
    
    for dep in sensor_deps:
        install_pip_package(dep)
    
    # 可选依赖
    print("\n4. 可选库（按需安装）:")
    optional_deps = [
        "picamera",                    # 树莓派摄像头
        "pybluez",                     # 蓝牙
        "pyspnego",                    # Windows认证
    ]
    
    for dep in optional_deps:
        install_pip_package(dep)


def setup_permissions(platform_info):
    """设置权限（Linux）"""
    if platform_info["system"] != "linux":
        return
    
    print("\n" + "="*60)
    print("设置硬件权限")
    print("="*60)
    
    if platform_info["is_raspberry_pi"]:
        print("\n树莓派权限设置:")
        
        # 启用接口
        commands = [
            "sudo raspi-config nonint do_i2c 0",  # 启用I2C
            "sudo raspi-config nonint do_spi 0",  # 启用SPI
            "sudo raspi-config nonint do_serial 0",  # 启用串口（硬件）
        ]
        
        for cmd in commands:
            print(f"执行: {cmd}")
            success, stdout, stderr = run_command(cmd)
            print(f"  {'✓' if success else '✗'}")
        
        # 添加用户到硬件组
        groups = ["gpio", "i2c", "spi", "dialout"]
        current_user = os.environ.get("USER", os.environ.get("USERNAME", ""))
        
        if current_user:
            for group in groups:
                print(f"添加用户到 {group} 组...")
                success, stdout, stderr = run_command(f"sudo usermod -a -G {group} {current_user}")
                print(f"  {'✓' if success else '✗'}")
    
    else:
        print("\nLinux通用权限设置:")
        
        # 串口权限
        print("设置串口权限...")
        success, stdout, stderr = run_command("sudo usermod -a -G dialout $USER")
        print(f"  {'✓' if success else '✗'}")
        
        # USB权限
        print("设置USB权限...")
        udev_rule = 'SUBSYSTEM=="usb", MODE="0666"'
        success, stdout, stderr = run_command(f'echo \'{udev_rule}\' | sudo tee /etc/udev/rules.d/99-usb-permissions.rules')
        if success:
            run_command("sudo udevadm control --reload-rules")
            run_command("sudo udevadm trigger")
            print("  ✓ USB权限规则已添加")
        else:
            print("  ✗ USB权限规则添加失败")


def verify_installation(platform_info):
    """验证安装"""
    print("\n" + "="*60)
    print("验证安装")
    print("="*60)
    
    test_modules = [
        ("串口", "serial"),
        ("USB", "usb"),
        ("GPIO (gpiozero)", "gpiozero"),
    ]
    
    if platform_info["system"] == "linux":
        if platform_info["is_raspberry_pi"]:
            test_modules.extend([
                ("GPIO (RPi.GPIO)", "RPi.GPIO"),
                ("I2C (smbus2)", "smbus2"),
                ("SPI (spidev)", "spidev"),
            ])
        else:
            test_modules.extend([
                ("I2C (smbus2)", "smbus2"),
                ("SPI (spidev)", "spidev"),
            ])
    
    all_ok = True
    
    for module_name, import_name in test_modules:
        print(f"检查 {module_name}...")
        success, stdout, stderr = run_command(
            f"{sys.executable} -c \"import {import_name.replace('-', '_')}; print('OK')\""
        )
        
        if success and "OK" in stdout:
            print(f"  ✓ {module_name} 可用")
        else:
            print(f"  ✗ {module_name} 不可用")
            all_ok = False
    
    return all_ok


def print_summary(platform_info, installation_ok):
    """打印安装总结"""
    print("\n" + "="*60)
    print("安装总结")
    print("="*60)
    
    print(f"操作系统: {platform_info['system'].upper()}")
    print(f"架构: {platform_info['machine']}")
    print(f"Python版本: {platform_info['python_version']}")
    
    if platform_info["is_raspberry_pi"]:
        print("平台: 树莓派")
    
    print(f"\n硬件依赖安装: {'✓ 完成' if installation_ok else '✗ 部分失败'}")
    
    if installation_ok:
        print("\n下一步:")
        print("  1. 如果是树莓派，请重启系统使权限生效")
        print("  2. 运行测试: python -m models.system_control.real_hardware.test_real_hardware")
        print("  3. 参考文档: models/system_control/real_hardware/hardware_dependencies.md")
    else:
        print("\n问题解决:")
        print("  1. 检查网络连接")
        print("  2. 确保有管理员/root权限")
        print("  3. 查看上方错误信息")
        print("  4. 手动安装缺失的依赖")
    
    print("\n注意:")
    print("  - 某些硬件可能需要额外的驱动程序")
    print("  - Windows用户可能需要安装Visual C++ Build Tools")
    print("  - 树莓派用户需要启用硬件接口")
    print("  - 如有问题，请查看硬件依赖文档")


def main():
    """主函数"""
    print("="*60)
    print("Self AGI 系统 - 硬件依赖安装工具")
    print("="*60)
    print("版本: 1.0.0")
    print("日期: 2026-03-29")
    print("作者: silencecrowtom@qq.com")
    print("="*60)
    
    # 获取平台信息
    platform_info = get_platform_info()
    print(f"\n检测到系统: {platform_info['system'].upper()} ({platform_info['machine']})")
    if platform_info["is_raspberry_pi"]:
        print("检测到树莓派")
    
    # 检查Python版本
    python_version = platform_info["python_version"]
    major, minor, _ = map(int, python_version.split('.'))
    if major < 3 or (major == 3 and minor < 8):
        print(f"警告: Python版本 {python_version} 可能过旧，建议使用Python 3.8+")
    
    # 安装系统包
    if platform_info["system"] == "windows":
        install_windows_dependencies()
    else:
        install_system_packages(platform_info)
    
    # 安装硬件依赖
    install_hardware_dependencies(platform_info)
    
    # 设置权限
    if platform_info["system"] == "linux":
        setup_permissions(platform_info)
    
    # 验证安装
    installation_ok = verify_installation(platform_info)
    
    # 打印总结
    print_summary(platform_info, installation_ok)
    
    # 退出代码
    return 0 if installation_ok else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n安装被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n安装过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)