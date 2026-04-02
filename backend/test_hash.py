#!/usr/bin/env python3
"""测试密码哈希算法一致性"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from passlib.context import CryptContext
import hashlib

# main.py中的算法
pwd_context_main = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

def get_password_hash_main(password: str) -> str:
    """main.py中的哈希算法"""
    sha256_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_context_main.hash(sha256_hash)

def verify_password_main(plain_password: str, hashed_password: str) -> bool:
    """main.py中的验证算法"""
    sha256_hash = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
    return pwd_context_main.verify(sha256_hash, hashed_password)

# create_admin.py中的算法（更新后）
pwd_context_admin = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

def get_password_hash_admin(password: str) -> str:
    """create_admin.py中的哈希算法（更新后）"""
    sha256_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_context_admin.hash(sha256_hash)

# core.security中的算法
pwd_context_core = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def get_password_hash_core(password: str) -> str:
    """core.security中的哈希算法"""
    return pwd_context_core.hash(password)

def verify_password_core(plain_password: str, hashed_password: str) -> bool:
    """core.security中的验证算法"""
    return pwd_context_core.verify(plain_password, hashed_password)

if __name__ == "__main__":
    password = "demopassword"
    
    print("测试密码哈希算法一致性")
    print(f"密码: {password}")
    print()
    
    # 生成各种哈希
    hash_main = get_password_hash_main(password)
    hash_admin = get_password_hash_admin(password)
    hash_core = get_password_hash_core(password)
    
    print(f"main.py哈希: {hash_main}")
    print(f"create_admin.py哈希: {hash_admin}")
    print(f"core.security哈希: {hash_core}")
    print()
    
    # 验证main.py算法
    print("验证main.py算法:")
    print(f"  hash_main验证: {verify_password_main(password, hash_main)}")
    print(f"  hash_admin验证: {verify_password_main(password, hash_admin)}")
    print(f"  hash_core验证: {verify_password_main(password, hash_core)}")
    print()
    
    # 验证core.security算法
    print("验证core.security算法:")
    print(f"  hash_main验证: {verify_password_core(password, hash_main)}")
    print(f"  hash_admin验证: {verify_password_core(password, hash_admin)}")
    print(f"  hash_core验证: {verify_password_core(password, hash_core)}")