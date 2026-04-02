# 真实硬件接口依赖说明

## 概述

本模块提供真实硬件接口支持，包括电机控制、传感器数据采集等。由于涉及真实的硬件通信，需要安装特定的硬件依赖库。

## 核心原则

- **禁止模拟，强制真实**：根据项目要求"禁止使用虚拟数据"，必须使用真实硬件，硬件不可用时抛出异常
- **模块化设计**：每个硬件类型有独立的接口实现
- **跨平台支持**：支持Windows、Linux、macOS等操作系统

## 依赖分类

### 1. 基础依赖（必需）

以下依赖是基础硬件接口库，建议全部安装：

```bash
# 串口通信
pip install pyserial

# I2C通信（Linux）
pip install smbus2

# SPI通信（Linux）
pip install spidev

# USB设备通信
pip install pyusb

# CAN总线通信
pip install python-can

# 网络通信（内置，无需安装）
# socket库是Python标准库
```

### 2. 电机控制依赖

根据电机类型和控制接口选择安装：

```bash
# GPIO控制（树莓派等）
pip install RPi.GPIO          # 树莓派原生GPIO
pip install gpiozero          # 跨平台GPIO库（推荐）
pip install Adafruit_PCA9685  # PWM扩展板

# 串口电机控制
pip install pyserial          # 已在上方安装

# 步进电机控制
pip install RPi.GPIO          # 步进电机驱动
```

### 3. 传感器依赖

根据传感器类型选择安装：

```bash
# 模拟传感器（ADC转换）
pip install adafruit-circuitpython-ads1x15  # ADS1115/ADS1015 ADC转换器

# I2C传感器通用
pip install smbus2                          # 已在上方安装

# 特定传感器库
pip install adafruit-circuitpython-dht      # DHT温湿度传感器
pip install adafruit-circuitpython-bmp280   # BMP280气压传感器
pip install adafruit-circuitpython-vl53l0x  # VL53L0X距离传感器

# 摄像头
pip install opencv-python                   # OpenCV
pip install picamera                        # 树莓派摄像头

# 蓝牙
pip install pybluez                         # Windows/Linux蓝牙
```

### 4. 机器人平台依赖

```bash
# NAOqi机器人
# 需要从Aldebaran官网下载SDK
# https://www.aldebaran.com/en/support/nao-6/downloads

# Arduino控制
pip install pyserial                        # 串口通信
# 需要安装Arduino IDE和相应库

# ROS（机器人操作系统）
# 需要安装ROS，参考：http://wiki.ros.org/
```

## 操作系统特定说明

### Windows

1. **串口设备**：
   - 安装CP210x或CH340等USB转串口驱动
   - 设备管理器查看COM端口号

2. **USB设备**：
   - 可能需要安装libusb驱动
   - 使用Zadig工具安装WinUSB驱动

3. **GPIO**：
   - Windows不支持直接GPIO
   - 需要真实GPIO硬件，禁止模拟模式

### Linux（树莓派等）

1. **启用接口**：
   ```bash
   # 启用I2C
   sudo raspi-config
   # Interface Options -> I2C -> Yes
   
   # 启用SPI
   sudo raspi-config
   # Interface Options -> SPI -> Yes
   
   # 启用串口
   sudo raspi-config
   # Interface Options -> Serial Port -> No (login shell) -> Yes (hardware)
   ```

2. **用户权限**：
   ```bash
   # 将用户加入gpio、i2c、spi组
   sudo usermod -a -G gpio,i2c,spi $USER
   ```

3. **重启生效**：
   ```bash
   sudo reboot
   ```

### macOS

1. **串口设备**：
   - 安装CP210x或CH340驱动
   - 设备路径：/dev/tty.usbserial-*

2. **权限设置**：
   ```bash
   # 修复串口权限
   sudo chmod 666 /dev/tty.usbserial-*
   ```

## 安装脚本

我们提供了自动安装脚本，可以一键安装常用依赖：

```bash
# 安装基础依赖
python -m models.system_control.real_hardware.install_dependencies
```

### install_dependencies.py 脚本

```python
#!/usr/bin/env python3
import subprocess
import sys

def install_dependencies():
    """安装硬件依赖"""
    
    dependencies = [
        "pyserial",      # 串口通信
        "smbus2",        # I2C通信
        "spidev",        # SPI通信
        "pyusb",         # USB设备
        "python-can",    # CAN总线
        "gpiozero",      # GPIO控制
        "Adafruit-PCA9685",  # PWM扩展板
    ]
    
    print("安装硬件依赖...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"  ✓ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ✗ {dep} 安装失败")
    
    print("\n安装完成！")
    print("注意：某些依赖可能需要系统级安装或硬件支持。")

if __name__ == "__main__":
    install_dependencies()
```

## 硬件配置示例

### 1. 电机控制器配置

```python
# 伺服电机（PWM控制）
motor_config = {
    "motor_type": "servo",
    "control_interface": "pwm",
    "pwm_pin": 18,
    "pwm_frequency": 50,  # Hz
    "pwm_min_duty": 2.5,   # % (0度)
    "pwm_max_duty": 12.5,  # % (180度)
    "max_speed": 1000.0,
    "max_torque": 5.0,
}

# 步进电机（GPIO控制）
stepper_config = {
    "motor_type": "stepper",
    "control_interface": "gpio",
    "direction_pin": 17,
    "step_pin": 18,
    "enable_pin": 27,
    "steps_per_revolution": 200,
    "max_speed": 500,  # 脉冲/秒
}
```

### 2. 传感器配置

```python
# 温度传感器（I2C）
temp_sensor_config = {
    "sensor_type": "temperature",
    "sensor_interface": "i2c",
    "bus": 1,
    "address": 0x40,
    "sampling_rate": 1.0,  # Hz
    "range_min": -40.0,
    "range_max": 125.0,
}

# 距离传感器（串口）
distance_sensor_config = {
    "sensor_type": "distance",
    "sensor_interface": "serial",
    "port": "/dev/ttyUSB0",
    "baudrate": 115200,
    "sampling_rate": 10.0,
}
```

## 故障排除

### 常见问题

1. **"No module named 'RPi.GPIO'"**
   - 原因：不在树莓派上运行或未安装RPi.GPIO
   - 解决方案：安装gpiozero（跨平台兼容）或确保在树莓派上运行

2. **串口权限错误**
   ```bash
   # Linux/Mac
   sudo chmod 666 /dev/ttyUSB0
   
   # 或添加用户到dialout组
   sudo usermod -a -G dialout $USER
   ```

3. **I2C设备未找到**
   ```bash
   # 检查I2C是否启用
   sudo i2cdetect -l
   
   # 扫描I2C设备
   sudo i2cdetect -y 1
   ```

4. **USB设备权限问题**
   ```bash
   # 创建udev规则
   echo 'SUBSYSTEM=="usb", MODE="0666"' | sudo tee /etc/udev/rules.d/99-usb-permissions.rules
   sudo udevadm control --reload-rules
   ```

### 模拟模式（已禁用）

根据项目要求"禁止使用虚拟数据"，模拟模式已被禁用：
- 电机控制器：必须使用真实硬件，硬件不可用时抛出异常
- 传感器：必须使用真实传感器，硬件不可用时抛出异常
- 日志中会显示"硬件不可用"错误

要确保硬件可用：
```python
# 必须安装所有硬件依赖库
# 必须确保硬件连接正常
# 硬件初始化失败将导致异常
```

## 测试硬件连接

使用测试脚本验证硬件连接：

```bash
# 运行硬件测试
python -m models.system_control.real_hardware.test_real_hardware

# 测试特定硬件
python -c "
from models.system_control.real_hardware import RealHardwareManager
manager = RealHardwareManager()
print('硬件管理器初始化成功')
"
```

## 更新日志

### v1.0.0 (2026-03-29)
- 初始版本
- 支持电机控制器（伺服、步进、直流、无刷电机）
- 支持传感器接口（温度、湿度、压力、距离等）
- 支持多种通信接口（PWM、GPIO、I2C、SPI、串口、USB、网络）
- 硬件管理器集成
- 模拟后备模式

## 支持与反馈

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 检查硬件连接和配置
3. 在项目Issue中反馈问题
4. 联系开发团队：silencecrowtom@qq.com

---

**注意**：使用真实硬件时，请确保：
1. 电源电压和电流符合硬件要求
2. 连接正确，避免短路
3. 遵循硬件制造商的安全指南
4. 在专业人员指导下操作高压/大电流设备