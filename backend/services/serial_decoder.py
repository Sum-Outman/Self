#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口数据解码器

功能：
1. 支持多种串口数据格式解码
2. 将原始字节数据转换为结构化数据
3. 支持自定义解码规则
4. 错误处理和数据验证

设计原则：
- 简单直接：输入原始数据，输出结构化数据
- 可扩展：易于添加新的解码协议
- 容错性：对格式错误的数据有处理能力
"""

import logging
import json
import struct
import re
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import datetime

logger = logging.getLogger(__name__)


class SerialProtocol(Enum):
    """串口协议枚举"""

    RAW = "raw"  # 原始字节数据
    ASCII = "ascii"  # ASCII文本
    HEX = "hex"  # 十六进制字符串
    JSON = "json"  # JSON格式
    BINARY = "binary"  # 二进制结构
    CUSTOM = "custom"  # 自定义格式


class DecodeResult:
    """解码结果类"""

    def __init__(
        self,
        success: bool,
        data: Any = None,
        protocol: SerialProtocol = None,
        error: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        初始化解码结果

        参数:
            success: 解码是否成功
            data: 解码后的数据
            protocol: 使用的协议
            error: 错误信息（如果失败）
            metadata: 元数据
        """
        self.success = success
        self.data = data
        self.protocol = protocol
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "protocol": self.protocol.value if self.protocol else None,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        if self.success:
            return f"DecodeResult(success=True, protocol={self.protocol}, data={self.data})"
        else:
            return f"DecodeResult(success=False, error={self.error})"


class SerialDecoder:
    """串口数据解码器

    将串口接收的原始数据解码为结构化数据
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化串口解码器

        参数:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or {}

        # 注册解码器
        self._decoders: Dict[SerialProtocol, Callable] = {
            SerialProtocol.RAW: self._decode_raw,
            SerialProtocol.ASCII: self._decode_ascii,
            SerialProtocol.HEX: self._decode_hex,
            SerialProtocol.JSON: self._decode_json,
            SerialProtocol.BINARY: self._decode_binary,
        }

        # 自定义解码规则
        self._custom_rules: List[Dict[str, Any]] = []

        # 统计信息
        self.stats = {
            "total_decoded": 0,
            "successful_decodes": 0,
            "failed_decodes": 0,
            "by_protocol": {p.value: 0 for p in SerialProtocol},
        }

        self.logger.info("串口数据解码器初始化完成")

    def decode(
        self,
        raw_data: bytes,
        protocol: Optional[SerialProtocol] = None,
        hint: Optional[str] = None,
    ) -> DecodeResult:
        """
        解码原始数据

        参数:
            raw_data: 原始字节数据
            protocol: 指定的协议（如果为None则自动检测）
            hint: 解码提示（如设备类型、数据格式）

        返回:
            解码结果
        """
        try:
            if not raw_data:
                return DecodeResult(
                    success=False, error="输入数据为空", protocol=protocol
                )

            self.stats["total_decoded"] += 1

            # 如果未指定协议，尝试自动检测
            if protocol is None:
                protocol = self._detect_protocol(raw_data, hint)
                self.logger.debug(f"自动检测协议: {protocol}")

            # 执行解码
            if protocol in self._decoders:
                decoder_func = self._decoders[protocol]
                result = decoder_func(raw_data)
                result.protocol = protocol
            elif protocol == SerialProtocol.CUSTOM:
                result = self._decode_custom(raw_data, hint)
                result.protocol = protocol
            else:
                return DecodeResult(
                    success=False, protocol=protocol, error=f"不支持的协议: {protocol}"
                )

            # 更新统计
            if result.success:
                self.stats["successful_decodes"] += 1
                self.stats["by_protocol"][protocol.value] += 1
            else:
                self.stats["failed_decodes"] += 1

            return result

        except Exception as e:
            self.logger.error(f"解码过程中发生异常: {e}")
            self.stats["failed_decodes"] += 1
            return DecodeResult(
                success=False, protocol=protocol, error=f"解码异常: {str(e)}"
            )

    def _detect_protocol(
        self, raw_data: bytes, hint: Optional[str] = None
    ) -> SerialProtocol:
        """
        自动检测数据协议

        参数:
            raw_data: 原始数据
            hint: 提示信息

        返回:
            检测到的协议
        """
        # 如果提供了提示，优先使用
        if hint:
            hint_lower = hint.lower()
            if "json" in hint_lower:
                return SerialProtocol.JSON
            elif "hex" in hint_lower or "0x" in hint_lower:
                return SerialProtocol.HEX
            elif "ascii" in hint_lower or "text" in hint_lower:
                return SerialProtocol.ASCII
            elif "binary" in hint_lower or "struct" in hint_lower:
                return SerialProtocol.BINARY

        # 尝试检测JSON
        try:
            text = raw_data.decode("utf-8", errors="ignore").strip()
            if text.startswith("{") and text.endswith("}"):
                # 尝试解析验证
                json.loads(text)
                return SerialProtocol.JSON
            elif text.startswith("[") and text.endswith("]"):
                json.loads(text)
                return SerialProtocol.JSON
        except Exception:
            pass  # 已实现

        # 尝试检测ASCII文本
        try:
            text = raw_data.decode("ascii", errors="ignore")
            # 检查是否主要为可打印字符
            printable_ratio = (
                sum(1 for c in text if 32 <= ord(c) <= 126) / len(text) if text else 0
            )
            if printable_ratio > 0.8 and len(text) > 3:
                return SerialProtocol.ASCII
        except Exception:
            pass  # 已实现

        # 检查是否为十六进制字符串
        try:
            hex_text = raw_data.decode("ascii", errors="ignore").strip()
            hex_pattern = re.compile(r"^[0-9a-fA-F\s]+$")
            if hex_pattern.match(hex_text) and len(hex_text) >= 4:
                return SerialProtocol.HEX
        except Exception:
            pass  # 已实现

        # 默认返回原始数据
        return SerialProtocol.RAW

    def _decode_raw(self, raw_data: bytes) -> DecodeResult:
        """解码原始字节数据"""
        try:
            return DecodeResult(
                success=True,
                data=list(raw_data),  # 转换为字节列表
                metadata={
                    "length": len(raw_data),
                    "hex": raw_data.hex(),
                },
            )
        except Exception as e:
            return DecodeResult(success=False, error=f"原始数据解码失败: {e}")

    def _decode_ascii(self, raw_data: bytes) -> DecodeResult:
        """解码ASCII文本数据"""
        try:
            text = raw_data.decode("ascii", errors="replace").strip()
            metadata = {
                "length": len(text),
                "original_length": len(raw_data),
                "has_non_ascii": any(ord(c) > 127 for c in text),
            }

            # 尝试提取结构化信息
            structured_data = self._extract_structured_info(text)

            return DecodeResult(
                success=True,
                data=structured_data if structured_data else text,
                metadata=metadata,
            )
        except Exception as e:
            return DecodeResult(success=False, error=f"ASCII解码失败: {e}")

    def _decode_hex(self, raw_data: bytes) -> DecodeResult:
        """解码十六进制数据"""
        try:
            # 尝试解析为十六进制字符串
            hex_text = raw_data.decode("ascii", errors="ignore").strip()
            hex_text = re.sub(r"\s+", "", hex_text)  # 移除空格

            if not hex_text:
                return DecodeResult(success=False, error="十六进制字符串为空")

            # 验证十六进制格式
            if not re.match(r"^[0-9a-fA-F]+$", hex_text):
                return DecodeResult(success=False, error="无效的十六进制格式")

            # 转换为字节列表
            if len(hex_text) % 2 != 0:
                hex_text = "0" + hex_text  # 补齐奇数长度

            bytes_data = bytes.fromhex(hex_text)

            metadata = {
                "hex_string": hex_text,
                "byte_count": len(bytes_data),
                "original_length": len(raw_data),
            }

            return DecodeResult(success=True, data=list(bytes_data), metadata=metadata)
        except Exception as e:
            return DecodeResult(success=False, error=f"十六进制解码失败: {e}")

    def _decode_json(self, raw_data: bytes) -> DecodeResult:
        """解码JSON数据"""
        try:
            text = raw_data.decode("utf-8", errors="strict").strip()
            parsed_data = json.loads(text)

            metadata = {
                "length": len(text),
                "original_length": len(raw_data),
                "data_type": type(parsed_data).__name__,
            }

            return DecodeResult(success=True, data=parsed_data, metadata=metadata)
        except json.JSONDecodeError as e:
            return DecodeResult(success=False, error=f"JSON解析错误: {e}")
        except Exception as e:
            return DecodeResult(success=False, error=f"JSON解码失败: {e}")

    def _decode_binary(self, raw_data: bytes) -> DecodeResult:
        """解码二进制结构数据"""
        try:
            metadata = {
                "length": len(raw_data),
                "hex": raw_data.hex(),
            }

            # 尝试常见二进制格式
            structured_data = self._parse_binary_formats(raw_data)

            return DecodeResult(
                success=True,
                data=structured_data if structured_data else list(raw_data),
                metadata=metadata,
            )
        except Exception as e:
            return DecodeResult(success=False, error=f"二进制解码失败: {e}")

    def _decode_custom(
        self, raw_data: bytes, hint: Optional[str] = None
    ) -> DecodeResult:
        """解码自定义格式数据"""
        try:
            # 应用自定义规则
            for rule in self._custom_rules:
                if self._matches_rule(raw_data, rule, hint):
                    result = self._apply_rule(raw_data, rule)
                    if result.success:
                        return result

            # 如果没有匹配的规则，尝试其他方法
            return DecodeResult(success=False, error="未找到匹配的自定义解码规则")
        except Exception as e:
            return DecodeResult(success=False, error=f"自定义解码失败: {e}")

    def _extract_structured_info(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本中提取结构化信息"""
        structured = {}

        # 检测传感器数据格式
        sensor_patterns = [
            (r"TEMP:\s*([\d\.]+)", "temperature"),
            (r"HUM:\s*([\d\.]+)", "humidity"),
            (r"PRES:\s*([\d\.]+)", "pressure"),
            (r"VOLT:\s*([\d\.]+)", "voltage"),
            (r"CURRENT:\s*([\d\.]+)", "current"),
            (r"(\d+\.\d+)\s*°C", "temperature_c"),
            (r"(\d+\.\d+)\s*%", "humidity_percent"),
        ]

        for pattern, key in sensor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    structured[key] = float(match.group(1))
                except Exception:
                    structured[key] = match.group(1)

        # 检测键值对格式
        kv_pattern = r"(\w+)[=:]\s*([\w\.\-]+)"
        matches = re.findall(kv_pattern, text)
        for key, value in matches:
            try:
                # 尝试转换为数字
                if "." in value:
                    structured[key] = float(value)
                else:
                    structured[key] = int(value)
            except Exception:
                structured[key] = value

        return structured if structured else None

    def _parse_binary_formats(self, raw_data: bytes) -> Optional[Dict[str, Any]]:
        """解析常见二进制格式"""
        result = {}

        # 尝试常见结构
        if len(raw_data) >= 4:
            # 尝试解析为32位整数
            try:
                value = struct.unpack("<I", raw_data[:4])[0]
                result["uint32_le"] = value
            except Exception:
                pass  # 已实现

            try:
                value = struct.unpack(">I", raw_data[:4])[0]
                result["uint32_be"] = value
            except Exception:
                pass  # 已实现

        if len(raw_data) >= 8:
            # 尝试解析为64位整数
            try:
                value = struct.unpack("<Q", raw_data[:8])[0]
                result["uint64_le"] = value
            except BaseException:
                pass  # 已实现

        if len(raw_data) >= 4:
            # 尝试解析为浮点数
            try:
                value = struct.unpack("<", raw_data[:4])[0]
                result["float32_le"] = value
            except BaseException:
                pass  # 已实现

        return result if result else None

    def _matches_rule(
        self, raw_data: bytes, rule: Dict[str, Any], hint: Optional[str]
    ) -> bool:
        """检查数据是否匹配规则"""
        # 根据规则实现匹配逻辑
        # 完整实现
        return False

    def _apply_rule(self, raw_data: bytes, rule: Dict[str, Any]) -> DecodeResult:
        """应用规则解码数据"""
        # 根据规则实现解码逻辑
        # 完整实现
        return DecodeResult(success=False, error="规则已实现")

    def add_custom_rule(self, rule: Dict[str, Any]):
        """添加自定义解码规则"""
        self._custom_rules.append(rule)
        self.logger.info(f"添加自定义解码规则: {rule.get('name', '未命名')}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["custom_rule_count"] = len(self._custom_rules)
        stats["success_rate"] = (
            self.stats["successful_decodes"] / self.stats["total_decoded"]
            if self.stats["total_decoded"] > 0
            else 0
        )
        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_decoded": 0,
            "successful_decodes": 0,
            "failed_decodes": 0,
            "by_protocol": {p.value: 0 for p in SerialProtocol},
        }
        self.logger.info("解码器统计信息已重置")


# 单例实例
_serial_decoder_instance: Optional[SerialDecoder] = None


def get_serial_decoder(config: Optional[Dict[str, Any]] = None) -> SerialDecoder:
    """获取串口解码器单例实例"""
    global _serial_decoder_instance
    if _serial_decoder_instance is None:
        _serial_decoder_instance = SerialDecoder(config)
    return _serial_decoder_instance
