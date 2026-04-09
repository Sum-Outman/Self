#!/usr/bin/env python3
"""验证数据库中的密码哈希"""

from passlib.context import CryptContext
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# main.py中的算法
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码（与main.py保持一致）"""
    sha256_hash = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
    return pwd_context.verify(sha256_hash, hashed_password)


# 从数据库获取的哈希
admin_hash = (
    "$5$rounds=535000$uLqye7juRVSTA2cn$9RDTnbR706ITWyjxeyLUVuFynnUmHoiGF5V2.E3PGm/"
)
demo_hash = (
    "$5$rounds=535000$Ags0mgsCz5m.YJpp$2.WAGuc3zd4LuEZpzOoqPB74t0Q2pFzequ7zfrxIPv8"
)

print("验证数据库中的密码哈希")
print("=" * 50)

# 测试admin
admin_password = "admin123"
print(f"admin密码: {admin_password}")
print(f"admin哈希: {admin_hash}")
print(f"验证结果: {verify_password(admin_password, admin_hash)}")
print()

# 测试demo
demo_password = "demopassword"
print(f"demo密码: {demo_password}")
print(f"demo哈希: {demo_hash}")
print(f"验证结果: {verify_password(demo_password, demo_hash)}")
print()


# 生成新的哈希进行比较
def get_password_hash(password: str) -> str:
    sha256_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_context.hash(sha256_hash)


print("生成新的哈希进行比较:")
print(f"admin新哈希: {get_password_hash(admin_password)}")
print(f"demo新哈希: {get_password_hash(demo_password)}")
