/*
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
    String command = Serial.readStringUntil('\n');
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
      Serial.print("0 0 0");  // 磁力计数据: MPU6050无磁力计硬件，如需真实磁力计数据请连接外部磁力计（如HMC5883L/QMC5883L）并实现读取逻辑
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
