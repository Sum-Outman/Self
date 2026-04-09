#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arduino串口硬件驱动示例
展示如何使用SerialInterface控制Arduino机器人

本示例提供了一个完整的Arduino机器人控制示例，包括：
1. Arduino固件通信协议
2. Python驱动实现
3. 硬件控制示例
4. 传感器数据读取示例

硬件要求：
- Arduino开发板（Uno, Mega, Nano等）
- 舵机或电机（用于关节控制）
- 电位器或编码器（用于位置反馈）
- 可选：IMU传感器、摄像头等

接线示例：
- 舵机信号线 -> Arduino PWM引脚
- 电位器中间引脚 -> Arduino模拟输入引脚
- 电源和接地
"""

import sys
import os
import time
import logging
from typing import Dict, Any, Optional, List

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from hardware.robot_controller import (
    HardwareInterface, RobotJoint, SensorType,
    JointState, IMUData, CameraData, SerialInterface
)

logger = logging.getLogger(__name__)


class ArduinoRobotInterface(SerialInterface):
    """Arduino机器人接口扩展
    
    扩展SerialInterface，添加Arduino特定的命令和功能
    """
    
    def __init__(self, port: str = "COM3", baudrate: int = 115200, 
                 simulation_mode: bool = False):
        """
        初始化Arduino机器人接口
        
        参数:
            port: 串口端口 (Windows: COM3, Linux: /dev/ttyUSB0)
            baudrate: 波特率 (建议115200)
            simulation_mode: 是否为模拟模式
        """
        # 检查模拟模式（根据项目要求"禁止使用虚拟数据"）
        if simulation_mode:
            raise RuntimeError(
                "Arduino机器人接口不支持模拟模式\n"
                "根据项目要求'禁止使用虚拟数据'，不允许使用模拟模式。\n"
                "根据项目要求'不采用任何降级处理，直接报错'，尝试使用模拟模式时直接报错。\n"
                "请连接真实Arduino硬件或使用物理仿真环境。"
            )
        
        super().__init__(port=port, baudrate=baudrate, 
                        simulation_mode=False)  # 强制为False
        self._interface_type = "arduino_robot"
        
        # Arduino特定配置
        self.joint_limits = {
            RobotJoint.L_SHOULDER_PITCH: (-90.0, 90.0),  # 度
            RobotJoint.L_ELBOW_YAW: (-45.0, 45.0),
            RobotJoint.R_SHOULDER_PITCH: (-90.0, 90.0),
            RobotJoint.R_ELBOW_YAW: (-45.0, 45.0),
        }
        
        # 关节到Arduino引脚的映射
        self.joint_pin_map = {
            RobotJoint.L_SHOULDER_PITCH: 9,   # PWM引脚
            RobotJoint.L_ELBOW_YAW: 10,
            RobotJoint.R_SHOULDER_PITCH: 11,
            RobotJoint.R_ELBOW_YAW: 12,
        }
        
        # 传感器引脚映射
        self.sensor_pin_map = {
            SensorType.JOINT_POSITION: {
                RobotJoint.L_SHOULDER_PITCH: "A0",
                RobotJoint.L_ELBOW_YAW: "A1",
                RobotJoint.R_SHOULDER_PITCH: "A2",
                RobotJoint.R_ELBOW_YAW: "A3",
            },
            SensorType.IMU: "I2C"  # IMU通常通过I2C连接
        }
        
        logger.info(f"初始化Arduino机器人接口，端口: {port}, 波特率: {baudrate}")
        
        # 模拟响应字典
        self._simulation_responses = {
            "VERSION": "FIRMWAREv1.0_SELF_AGI_ARDUINO",
            "SET_SERVO_FREQ 50": "OK",
            "ENABLE_SENSORS": "OK",
            "DISABLE_SENSORS": "OK",
        }
    
    def send_command(self, command: str) -> Optional[str]:
        """发送命令到Arduino（支持模拟模式）"""
        if self._simulation_mode:
            # 模拟模式：返回预设响应或生成模拟响应
            if command in self._simulation_responses:
                return self._simulation_responses[command]
            
            # 生成模拟响应
            if command.startswith("SET_SERVO"):
                # 格式: SET_SERVO <pin> <angle>
                return "OK"
            elif command.startswith("READ_ANALOG"):
                # 格式: READ_ANALOG <pin>
                # 返回随机模拟值
                import random
                pin = command.split()[1] if len(command.split()) > 1 else "A0"
                value = random.randint(0, 1023)
                return f"ANALOG {pin} {value}"
            elif command == "READ_IMU":
                # 返回模拟IMU数据
                import random
                return f"IMU {random.uniform(-1, 1):.3f} {random.uniform(-1, 1):.3f} {9.8+random.uniform(-0.1, 0.1):.3f} {random.uniform(-0.1, 0.1):.3f} {random.uniform(-0.1, 0.1):.3f} {random.uniform(-0.1, 0.1):.3f} 0 0 0"
            elif command == "GET_JOINTS":
                # 返回模拟关节状态
                return "ALL_JOINTS\nL_SHOULDER_PITCH 45.0 0.0 0.0 25.0 12.0 0.5\nR_SHOULDER_PITCH -45.0 0.0 0.0 25.0 12.0 0.5"
            else:
                return "OK"
        
        # 真实模式：调用父类方法
        return super().send_command(command)
    
    def connect(self) -> bool:
        """连接到Arduino"""
        if self._simulation_mode:
            logger.info("Arduino模拟模式连接成功")
            # 在模拟模式下，我们需要设置连接状态
            self._simulation_connected = True
            # 模拟初始化
            self._initialize_arduino()
            return True
        
        # 调用父类的connect方法
        connected = super().connect()
        
        if connected:
            # 发送初始化命令
            self._initialize_arduino()
            
        return connected
    
    def is_connected(self) -> bool:
        """检查Arduino连接"""
        if self._simulation_mode:
            # 模拟模式下的连接状态
            return getattr(self, '_simulation_connected', False)
        
        # 真实模式：调用父类方法
        return super().is_connected()
    
    def disconnect(self) -> bool:
        """断开Arduino连接"""
        if self._simulation_mode:
            if getattr(self, '_simulation_connected', False):
                self._simulation_connected = False
                logger.info("Arduino模拟模式断开连接成功")
                return True
            return False
        
        # 真实模式：调用父类方法
        return super().disconnect()
    
    def _initialize_arduino(self):
        """初始化Arduino固件"""
        try:
            # 发送版本查询命令
            response = self.send_command("VERSION")
            if response:
                logger.info(f"Arduino固件版本: {response}")
            
            # 设置舵机频率
            self.send_command("SET_SERVO_FREQ 50")  # 50Hz标准舵机频率
            
            # 启用传感器
            self.send_command("ENABLE_SENSORS")
            
            logger.info("Arduino初始化完成")
            
        except Exception as e:
            logger.warning(f"Arduino初始化失败: {e}")
    
    def set_joint_position(self, joint: RobotJoint, position: float) -> bool:
        """设置关节位置（Arduino特定实现）
        
        参数:
            joint: 关节枚举
            position: 位置（角度，-180到180度）
            
        返回:
            成功返回True，失败返回False
        """
        # 检查关节限制
        if joint in self.joint_limits:
            min_limit, max_limit = self.joint_limits[joint]
            if position < min_limit or position > max_limit:
                logger.warning(f"关节 {joint.value} 位置 {position} 超出限制 [{min_limit}, {max_limit}]")
                position = max(min_limit, min(max_limit, position))
        
        # 获取引脚号
        if joint not in self.joint_pin_map:
            logger.error(f"关节 {joint.value} 未配置引脚映射")
            return False
        
        pin = self.joint_pin_map[joint]
        
        # 发送命令到Arduino
        command = f"SET_SERVO {pin} {position:.1f}"
        response = self.send_command(command)
        
        if response == "OK":
            logger.debug(f"设置关节 {joint.value} 到位置 {position} 度")
            return True
        else:
            logger.error(f"设置关节位置失败: {response}")
            return False
    
    def get_joint_position(self, joint: RobotJoint) -> Optional[float]:
        """获取关节位置（通过电位器或编码器）"""
        if joint not in self.sensor_pin_map[SensorType.JOINT_POSITION]:
            logger.error(f"关节 {joint.value} 未配置位置传感器")
            return None  # 返回None
        
        pin = self.sensor_pin_map[SensorType.JOINT_POSITION][joint]
        command = f"READ_ANALOG {pin}"
        response = self.send_command(command)
        
        if response and response.startswith("ANALOG"):
            try:
                # 解析响应: "ANALOG <pin> <value>"
                parts = response.split()
                if len(parts) >= 3:
                    analog_value = int(parts[2])
                    # 将模拟值转换为角度（假设0-1023对应-90到90度）
                    angle = (analog_value / 1023.0) * 180.0 - 90.0
                    return angle
            except ValueError as e:
                logger.error(f"解析关节位置失败: {e}")
        
        return None  # 返回None
    
    def read_imu_data(self) -> Optional[IMUData]:
        """读取IMU数据"""
        if not self.sensor_enabled:
            logger.warning("传感器功能已禁用")
            return None  # 返回None
        
        command = "READ_IMU"
        response = self.send_command(command)
        
        if response and response.startswith("IMU"):
            try:
                # 解析响应: "IMU <ax> <ay> <az> <gx> <gy> <gz> <mx> <my> <mz>"
                parts = response.split()
                if len(parts) >= 10:
                    import numpy as np
                    timestamp = time.time()
                    
                    return IMUData(
                        acceleration=np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                        gyroscope=np.array([float(parts[4]), float(parts[5]), float(parts[6])]),
                        magnetometer=np.array([float(parts[7]), float(parts[8]), float(parts[9])]),
                        orientation=np.zeros(3),  # 需要计算
                        timestamp=timestamp
                    )
            except (ValueError, IndexError) as e:
                logger.error(f"解析IMU数据失败: {e}")
        
        return None  # 返回None
    
    def perform_safe_movement(self, positions: Dict[RobotJoint, float], 
                             speed: float = 30.0) -> bool:
        """执行安全运动（逐步移动到目标位置）
        
        参数:
            positions: 目标位置字典
            speed: 运动速度（度/秒）
            
        返回:
            成功返回True
        """
        if not self.is_connected():
            logger.error("未连接到Arduino")
            return False
        
        logger.info(f"开始安全运动，速度: {speed} 度/秒")
        
        # 获取当前位置
        current_positions = {}
        for joint in positions.keys():
            pos = self.get_joint_position(joint)
            if pos is not None:
                current_positions[joint] = pos
            else:
                logger.warning(f"无法获取关节 {joint.value} 的当前位置，使用默认值0")
                current_positions[joint] = 0.0
        
        # 逐步移动到目标位置
        max_steps = 100
        step_delay = 0.05  # 50ms
        
        for step in range(max_steps):
            # 计算插值位置
            alpha = step / max_steps
            intermediate_positions = {}
            
            for joint, target_pos in positions.items():
                current_pos = current_positions.get(joint, 0.0)
                interp_pos = current_pos + (target_pos - current_pos) * alpha
                intermediate_positions[joint] = interp_pos
            
            # 设置中间位置
            success = self.set_joint_positions(intermediate_positions)
            
            if not success:
                logger.error(f"运动步骤 {step} 失败")
                return False
            
            time.sleep(step_delay)
        
        # 最终位置
        success = self.set_joint_positions(positions)
        
        if success:
            logger.info("安全运动完成")
        else:
            logger.error("安全运动失败")
        
        return success


def run_arduino_example():
    """运行Arduino示例"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Arduino串口硬件驱动示例 ===")
    
    # 创建Arduino接口
    # 注意：根据实际串口修改端口
    # Windows: COM3, COM4 等
    # Linux: /dev/ttyUSB0, /dev/ttyACM0 等
    
    port = "COM3"  # 修改为实际端口
    # port = "/dev/ttyUSB0"  # Linux示例
    
    # 使用模拟模式（如果没有真实硬件）
    simulation_mode = True
    
    arduino = ArduinoRobotInterface(
        port=port,
        baudrate=115200,
        simulation_mode=simulation_mode
    )
    
    try:
        # 连接Arduino
        print(f"连接到Arduino (端口: {port}, 模拟模式: {simulation_mode})...")
        if not arduino.connect():
            print("连接失败，退出示例")
            return
        
        print("连接成功!")
        
        # 获取接口信息
        info = arduino.get_interface_info()
        print(f"接口信息: {info}")
        
        # 示例1: 设置单个关节位置
        print("\n--- 示例1: 设置单个关节位置 ---")
        joint = RobotJoint.L_SHOULDER_PITCH
        position = 45.0  # 45度
        
        print(f"设置关节 {joint.value} 到 {position} 度")
        success = arduino.set_joint_position(joint, position)
        print(f"设置结果: {'成功' if success else '失败'}")
        
        # 等待运动完成
        time.sleep(1.0)
        
        # 示例2: 读取关节位置
        print("\n--- 示例2: 读取关节位置 ---")
        current_pos = arduino.get_joint_position(joint)
        if current_pos is not None:
            print(f"关节 {joint.value} 当前位置: {current_pos:.1f} 度")
        else:
            print("无法读取关节位置")
        
        # 示例3: 安全运动
        print("\n--- 示例3: 安全运动 ---")
        target_positions = {
            RobotJoint.L_SHOULDER_PITCH: -30.0,
            RobotJoint.L_ELBOW_YAW: 20.0,
            RobotJoint.R_SHOULDER_PITCH: 30.0,
            RobotJoint.R_ELBOW_YAW: -20.0,
        }
        
        print(f"目标位置: {target_positions}")
        success = arduino.perform_safe_movement(target_positions, speed=20.0)
        print(f"安全运动: {'成功' if success else '失败'}")
        
        # 示例4: 读取IMU数据
        print("\n--- 示例4: 读取IMU数据 ---")
        if arduino.sensor_enabled:
            imu_data = arduino.read_imu_data()
            if imu_data:
                print(f"加速度: {imu_data.acceleration}")
                print(f"陀螺仪: {imu_data.gyroscope}")
                print(f"磁力计: {imu_data.magnetometer}")
            else:
                print("无法读取IMU数据")
        else:
            print("传感器功能已禁用")
        
        # 示例5: 断开连接
        print("\n--- 示例5: 断开连接 ---")
        arduino.disconnect()
        print("Arduino已断开连接")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 示例完成 ===")


def generate_arduino_firmware_example():
    """生成Arduino固件示例代码"""
    
    firmware_code = """/*
 * Self AGI Arduino机器人固件
 * 与Python SerialInterface通信
 * 
 * 命令格式:
 * - SET_SERVO <pin> <angle>: 设置舵机角度
 * - READ_ANALOG <pin>: 读取模拟输入
 * - READ_IMU: 读取IMU数据
 * - VERSION: 获取固件版本
 * - ENABLE_SENSORS: 启用传感器
 * - DISABLE_SENSORS: 禁用传感器
 */

#include <Servo.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// 配置
#define BAUD_RATE 115200
#define SERVO_FREQ 50  // 舵机频率 (Hz)

// 舵机对象
Servo servos[4];
int servoPins[4] = {9, 10, 11, 12};  // 舵机引脚
float servoAngles[4] = {90, 90, 90, 90};  // 初始角度 (90度为中心)

// IMU传感器
Adafruit_MPU6050 mpu;
bool imuAvailable = false;

// 传感器启用状态
bool sensorsEnabled = true;

void setup() {
  Serial.begin(BAUD_RATE);
  
  // 初始化舵机
  for (int i = 0; i < 4; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(servoAngles[i]);
  }
  
  // 初始化IMU
  if (mpu.begin()) {
    imuAvailable = true;
    // 配置IMU
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }
  
  // 等待串口连接
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("READY: Self AGI Arduino Robot Firmware v1.0");
}

void loop() {
  // 检查串口命令
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\\n');
    command.trim();
    processCommand(command);
  }
  
  // 定期发送传感器数据
  static unsigned long lastSensorUpdate = 0;
  if (sensorsEnabled && millis() - lastSensorUpdate > 100) {  // 10Hz
    sendSensorData();
    lastSensorUpdate = millis();
  }
}

void processCommand(String cmd) {
  if (cmd.length() == 0) return;
  
  // 分割命令
  int spaceIndex = cmd.indexOf(' ');
  String cmdType = cmd;
  String params = "";
  
  if (spaceIndex != -1) {
    cmdType = cmd.substring(0, spaceIndex);
    params = cmd.substring(spaceIndex + 1);
  }
  
  cmdType.toUpperCase();
  
  if (cmdType == "VERSION") {
    Serial.println("FIRMWAREv1.0_SELF_AGI_ARDUINO");
    
  } else if (cmdType == "SET_SERVO") {
    // 格式: SET_SERVO <pin> <angle>
    int pin = params.substring(0, params.indexOf(' ')).toInt();
    float angle = params.substring(params.indexOf(' ') + 1).toFloat();
    
    int servoIndex = -1;
    for (int i = 0; i < 4; i++) {
      if (servoPins[i] == pin) {
        servoIndex = i;
        break;
      }
    }
    
    if (servoIndex != -1 && angle >= 0 && angle <= 180) {
      servos[servoIndex].write(angle);
      servoAngles[servoIndex] = angle;
      Serial.println("OK");
    } else {
      Serial.println("ERROR: Invalid pin or angle");
    }
    
  } else if (cmdType == "READ_ANALOG") {
    // 格式: READ_ANALOG <pin>
    int pin = params.toInt();
    if (pin >= A0 && pin <= A7) {
      int value = analogRead(pin);
      Serial.print("ANALOG ");
      Serial.print(pin);
      Serial.print(" ");
      Serial.println(value);
    } else {
      Serial.println("ERROR: Invalid analog pin");
    }
    
  } else if (cmdType == "READ_IMU") {
    if (imuAvailable) {
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);
      
      Serial.print("IMU ");
      Serial.print(a.acceleration.x); Serial.print(" ");
      Serial.print(a.acceleration.y); Serial.print(" ");
      Serial.print(a.acceleration.z); Serial.print(" ");
      Serial.print(g.gyro.x); Serial.print(" ");
      Serial.print(g.gyro.y); Serial.print(" ");
      Serial.print(g.gyro.z); Serial.print(" ");
      Serial.print("0 0 0");  // 磁力计实现
      Serial.println();
    } else {
      Serial.println("ERROR: IMU not available");
    }
    
  } else if (cmdType == "ENABLE_SENSORS") {
    sensorsEnabled = true;
    Serial.println("OK");
    
  } else if (cmdType == "DISABLE_SENSORS") {
    sensorsEnabled = false;
    Serial.println("OK");
    
  } else if (cmdType == "SET_SERVO_FREQ") {
    // 注意: 标准Arduino Servo库不支持频率设置
    // 这里仅作为示例
    Serial.println("OK: Frequency set (not implemented)");
    
  } else {
    Serial.print("ERROR: Unknown command: ");
    Serial.println(cmdType);
  }
}

void sendSensorData() {
  // 发送所有舵机角度
  Serial.print("JOINTS ");
  for (int i = 0; i < 4; i++) {
    Serial.print(servoAngles[i]);
    if (i < 3) Serial.print(" ");
  }
  Serial.println();
}
"""
    
    print("=== Arduino固件示例代码 ===")
    print("将以下代码上传到Arduino开发板：")
    print("\n" + firmware_code)
    
    # 保存到文件
    firmware_path = os.path.join(os.path.dirname(__file__), "arduino_firmware_example.ino")
    try:
        with open(firmware_path, "w", encoding="utf-8") as f:
            f.write(firmware_code)
        print(f"\n固件代码已保存到: {firmware_path}")
    except Exception as e:
        print(f"保存固件代码失败: {e}")


if __name__ == "__main__":
    # 运行Python示例
    run_arduino_example()
    
    # 生成Arduino固件代码
    print("\n" + "="*50)
    generate_arduino_firmware_example()