"""
测试系统控制接口

功能：
- 测试串口控制器
- 测试硬件管理器
- 测试传感器接口
- 测试电机控制器
- 测试系统监控器
"""

import sys
import os
import time
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_serial_controller():
    """测试串口控制器"""
    logger.info("测试串口控制器...")

    try:
        from models.system_control.serial_controller import (
            SerialController,
            SerialProtocol,
        )

        # 创建串口控制器
        controller = SerialController()

        # 列出可用串口
        ports = controller.list_available_ports()
        logger.info(f"发现 {len(ports)} 个可用串口")

        # 测试串口连接（仅测试，不实际连接）
        test_port = None
        for port_info in ports:
            logger.info(f"  串口: {port_info['device']} - {port_info['description']}")
            test_port = port_info["device"]
            break

        if test_port:
            # 测试串口可用性检查
            is_available = controller.is_port_available(test_port)
            logger.info(f"串口可用性检查: {test_port} -> {is_available}")

        # 测试数据编码/解码
        test_data = "Hello, Serial!"
        encoded = controller._encode_data(test_data, SerialProtocol.ASCII)
        decoded = controller._decode_data(encoded, SerialProtocol.ASCII)

        logger.info(f"数据编码/解码测试: '{test_data}' -> '{decoded}'")

        # 获取统计信息
        stats = controller.get_stats()
        logger.info(f"串口控制器统计: {stats}")

        # 断开连接（如果已连接）
        controller.disconnect()

        logger.info("✓ 串口控制器测试通过")
        return True

    except Exception as e:
        logger.error(f"串口控制器测试失败: {e}")
        return False


def test_hardware_manager():
    """测试硬件管理器"""
    logger.info("测试硬件管理器...")

    try:
        from models.system_control.hardware_manager import (
            HardwareManager,
            HardwareDevice,
            DeviceType,
            DeviceStatus,
        )

        # 创建硬件管理器
        manager = HardwareManager()

        # 启动硬件管理器
        manager.start()

        # 创建测试设备
        test_device = HardwareDevice(
            device_id="test_device_001",
            device_type=DeviceType.SENSOR,
            name="测试传感器",
            description="用于测试的虚拟传感器",
            manufacturer="Test Corp",
            model="TS-001",
            serial_number="SN001",
            version="1.0",
            connection_type="virtual",
            status=DeviceStatus.ONLINE,
            last_seen=time.time(),
            capabilities=["temperature_measurement", "humidity_measurement"],
            properties={"range": "-40~125°C", "accuracy": "±0.5°C"},
        )

        # 注册设备
        success = manager.register_device(test_device)
        logger.info(f"设备注册: {test_device.name} -> {success}")

        # 获取设备
        device = manager.get_device("test_device_001")
        if device:
            logger.info(f"获取设备成功: {device.name} ({device.device_type.value})")

        # 更新设备状态
        success = manager.update_device_status(
            "test_device_001", DeviceStatus.BUSY, "正在采集数据"
        )
        logger.info(f"设备状态更新: {success}")

        # 获取设备列表
        devices = manager.get_all_devices()
        logger.info(f"设备总数: {len(devices)}")

        # 按类型获取设备
        sensor_devices = manager.get_devices_by_type(DeviceType.SENSOR)
        logger.info(f"传感器设备数: {len(sensor_devices)}")

        # 发现设备
        discovered_devices = manager.discover_devices()
        logger.info(f"发现新设备数: {len(discovered_devices)}")

        # 获取统计信息
        stats = manager.get_stats()
        logger.info(f"硬件管理器统计: {stats}")

        # 停止硬件管理器
        manager.stop()

        logger.info("✓ 硬件管理器测试通过")
        return True

    except Exception as e:
        logger.error(f"硬件管理器测试失败: {e}")
        return False


def test_sensor_interface():
    """测试传感器接口"""
    logger.info("测试传感器接口...")

    try:
        from models.system_control.sensor_interface import (
            SensorInterface,
            SensorConfig,
            SensorType,
        )

        # 创建传感器接口
        sensor_interface = SensorInterface()

        # 创建传感器配置
        temperature_config = SensorConfig(
            sensor_id="temp_sensor_001",
            sensor_type=SensorType.TEMPERATURE,
            name="温度传感器",
            description="虚拟温度传感器",
            sampling_rate=1.0,
            sampling_interval=1.0,
            buffer_size=50,
            enable_filtering=True,
            enable_calibration=True,
            filter_type="moving_average",
            filter_window=5,
            calibration_params={"a": 1.0, "b": 0.0},
        )

        humidity_config = SensorConfig(
            sensor_id="humidity_sensor_001",
            sensor_type=SensorType.HUMIDITY,
            name="湿度传感器",
            description="虚拟湿度传感器",
            sampling_rate=0.5,
            sampling_interval=2.0,
            buffer_size=30,
            enable_filtering=True,
            enable_calibration=False,
        )

        # 注册传感器
        success1 = sensor_interface.register_sensor(temperature_config)
        success2 = sensor_interface.register_sensor(humidity_config)

        logger.info(f"温度传感器注册: {success1}")
        logger.info(f"湿度传感器注册: {success2}")

        # 启动传感器接口
        sensor_interface.start()

        # 等待传感器采集数据
        logger.info("等待传感器数据采集...")
        time.sleep(3)

        # 获取传感器数据
        temp_data = sensor_interface.read_sensor_data("temp_sensor_001")
        humidity_data = sensor_interface.read_sensor_data("humidity_sensor_001")

        if temp_data:
            logger.info(f"温度传感器数据: {temp_data.data} {temp_data.unit}")

        if humidity_data:
            logger.info(f"湿度传感器数据: {humidity_data.data} {humidity_data.unit}")

        # 获取数据历史
        temp_history = sensor_interface.get_sensor_data_history(
            "temp_sensor_001", limit=5
        )
        logger.info(f"温度数据历史记录数: {len(temp_history)}")

        # 获取传感器统计信息
        temp_stats = sensor_interface.get_sensor_stats("temp_sensor_001")
        if temp_stats:
            logger.info(f"温度传感器统计: {temp_stats.get('data_count', 0)} 个数据点")

        # 获取接口统计信息
        stats = sensor_interface.get_stats()
        logger.info(f"传感器接口统计: {stats}")

        # 停止传感器接口
        sensor_interface.stop()

        # 注销传感器
        sensor_interface.unregister_sensor("temp_sensor_001")
        sensor_interface.unregister_sensor("humidity_sensor_001")

        logger.info("✓ 传感器接口测试通过")
        return True

    except Exception as e:
        logger.error(f"传感器接口测试失败: {e}")
        return False


def test_motor_controller():
    """测试电机控制器"""
    logger.info("测试电机控制器...")

    try:
        from models.system_control.motor_controller import (
            MotorController,
            MotorConfig,
            MotorType,
            MotorControlMode,
        )

        # 创建电机控制器
        controller = MotorController()

        # 创建电机配置
        motor_config = MotorConfig(
            motor_id="test_motor_001",
            motor_type=MotorType.SERVO,
            name="测试伺服电机",
            description="用于测试的虚拟伺服电机",
            max_position=180.0,
            min_position=0.0,
            max_velocity=100.0,
            max_acceleration=50.0,
            max_torque=2.0,
            max_current=1.5,
            max_voltage=12.0,
            control_mode=MotorControlMode.POSITION,
            pid_params={"kp": 1.0, "ki": 0.1, "kd": 0.01},
            deadband=0.5,
            connection_type="virtual",
            overheat_threshold=80.0,
            overload_threshold=0.9,
            stall_threshold=0.1,
        )

        # 注册电机
        success = controller.register_motor(motor_config)
        logger.info(f"电机注册: {motor_config.name} -> {success}")

        # 启动电机控制器
        controller.start()

        # 等待控制器初始化
        time.sleep(1)

        # 获取电机状态
        state = controller.get_motor_state("test_motor_001")
        if state:
            logger.info(
                f"电机初始状态: 位置={state.position:.2f}, 速度={state.velocity:.2f}"
            )

        # 发送移动命令
        logger.info("发送移动命令: 位置=90.0, 速度因子=0.5")
        success = controller.move_to_position(
            motor_id="test_motor_001", position=90.0, speed_factor=0.5, blocking=False
        )

        logger.info(f"移动命令发送: {success}")

        # 等待电机移动
        logger.info("等待电机移动...")
        time.sleep(3)

        # 获取更新后的状态
        state = controller.get_motor_state("test_motor_001")
        if state:
            logger.info(
                f"电机移动后状态: 位置={state.position:.2f}, "
                f"速度={state.velocity:.2f}, 状态={state.status.value}"
            )

        # 发送速度命令
        logger.info("发送速度命令: 速度=30.0, 持续时间=2.0")
        success = controller.set_velocity(
            motor_id="test_motor_001",
            velocity=30.0,
            duration=2.0,
            blocking=False,
        )

        logger.info(f"速度命令发送: {success}")

        # 等待速度控制
        time.sleep(3)

        # 停止电机
        logger.info("停止电机...")
        controller.stop_motor("test_motor_001")

        # 获取停止后状态
        state = controller.get_motor_state("test_motor_001")
        if state:
            logger.info(
                f"电机停止后状态: 位置={state.position:.2f}, "
                f"速度={state.velocity:.2f}, 状态={state.status.value}"
            )

        # 获取所有电机状态
        all_states = controller.get_all_motor_states()
        logger.info(f"所有电机状态数: {len(all_states)}")

        # 获取统计信息
        stats = controller.get_stats()
        logger.info(f"电机控制器统计: {stats}")

        # 停止电机控制器
        controller.stop()

        # 注销电机
        controller.unregister_motor("test_motor_001")

        logger.info("✓ 电机控制器测试通过")
        return True

    except Exception as e:
        logger.error(f"电机控制器测试失败: {e}")
        return False


def test_system_monitor():
    """测试系统监控器"""
    logger.info("测试系统监控器...")

    try:
        from models.system_control.system_monitor import SystemMonitor

        # 创建系统监控器
        monitor = SystemMonitor(
            {
                "monitoring_interval": 2.0,
                "metrics_history_size": 10,
                "alerts_history_size": 10,
                "cpu_threshold_warning": 50.0,
                "cpu_threshold_error": 80.0,
                "memory_threshold_warning": 70.0,
                "memory_threshold_error": 90.0,
            }
        )

        # 启动系统监控器
        monitor.start()

        # 等待监控器收集数据
        logger.info("等待系统监控器收集数据...")
        time.sleep(5)

        # 获取系统信息
        system_info = monitor.get_system_info()
        logger.info(f"系统信息: {system_info.get('platform', 'Unknown')}")
        logger.info(f"CPU核心数: {system_info.get('cpu_count', 'N/A')}")
        logger.info(f"总内存: {system_info.get('memory_total_gb', 'N/A'):.2f} GB")

        # 获取当前指标
        current_metrics = monitor.get_current_metrics()
        logger.info(f"当前指标数: {len(current_metrics)}")

        # 显示关键指标
        for metric in current_metrics:
            if metric.metric_id in ["cpu_usage", "memory_usage", "disk_usage_C_"]:
                logger.info(
                    f"  {metric.name}: {metric.value:.1f}{metric.unit} "
                    f"({metric.status.value})"
                )

        # 获取指标历史
        cpu_history = monitor.get_metric_history("cpu_usage", limit=5)
        logger.info(f"CPU使用率历史记录数: {len(cpu_history)}")

        # 获取活跃警报
        active_alerts = monitor.get_active_alerts()
        logger.info(f"活跃警报数: {len(active_alerts)}")

        # 获取警报历史
        alerts_history = monitor.get_alerts_history(limit=5)
        logger.info(f"警报历史记录数: {len(alerts_history)}")

        # 获取统计信息
        stats = monitor.get_stats()
        logger.info(f"系统监控器统计: {stats}")

        # 停止系统监控器
        monitor.stop()

        logger.info("✓ 系统监控器测试通过")
        return True

    except Exception as e:
        logger.error(f"系统监控器测试失败: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("开始测试系统控制接口")

    results = []

    # 测试串口控制器
    results.append(("串口控制器", test_serial_controller()))

    # 测试硬件管理器
    results.append(("硬件管理器", test_hardware_manager()))

    # 测试传感器接口
    results.append(("传感器接口", test_sensor_interface()))

    # 测试电机控制器
    results.append(("电机控制器", test_motor_controller()))

    # 测试系统监控器
    results.append(("系统监控器", test_system_monitor()))

    # 输出测试结果
    logger.info("\n" + "=" * 50)
    logger.info("测试结果汇总:")

    all_passed = True
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 50)

    if all_passed:
        logger.info("✅ 所有系统控制接口测试通过！")
        return True
    else:
        logger.error("❌ 部分系统控制接口测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
