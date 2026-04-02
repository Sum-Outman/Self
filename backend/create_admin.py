#!/usr/bin/env python3
"""
创建默认管理员用户脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from passlib.context import CryptContext
import uuid
from datetime import datetime, timezone
import hashlib

# 导入数据库模型
from backend.db_models.chat import ChatSession, ChatMessage  # 先导入Chat模型以解决循环依赖
from backend.db_models.user import User
from backend.core.database import SessionLocal, Base, engine

# 创建密码上下文 - 与main.py保持一致（SHA-256预哈希 + sha256_crypt）
import hashlib
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# 复制main.py中的密码哈希逻辑
def get_password_hash(password: str) -> str:
    """获取密码哈希，与main.py保持一致"""
    # 使用SHA-256预哈希处理长密码问题（与main.py相同）
    sha256_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_context.hash(sha256_hash)

def create_admin_user():
    """创建默认管理员用户"""
    # 确保数据库表存在
    Base.metadata.create_all(bind=engine)
    
    # 从环境变量获取密码，如果不存在则使用默认值（开发环境）
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@selfagi.com")
    
    # 检查密码强度（最小长度8字符）
    if len(admin_password) < 8:
        print(f"警告：管理员密码长度不足（{len(admin_password)}字符），建议至少8字符")
    
    db = SessionLocal()
    try:
        # 检查是否已存在管理员用户
        existing_admin = db.query(User).filter(User.username == "admin").first()
        if existing_admin:
            print("管理员用户已存在，更新密码和邮箱...")
            # 更新密码为新哈希
            existing_admin.hashed_password = get_password_hash(admin_password)
            existing_admin.email = admin_email
            existing_admin.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(existing_admin)
            print("管理员用户信息已更新")
            return existing_admin
        
        # 创建新管理员用户
        admin_user = User(
            username="admin",
            email=admin_email,
            hashed_password=get_password_hash(admin_password),
            full_name="系统管理员",
            is_admin=True,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            last_login=None
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print("管理员用户创建成功")
        print(f"用户名: admin")
        print(f"邮箱: {admin_email}")
        print("提示：生产环境请通过环境变量ADMIN_PASSWORD设置安全密码")
        
        return admin_user
    except Exception as e:
        print(f"创建管理员用户失败: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_demo_user():
    """创建演示用户"""
    # 从环境变量获取密码，如果不存在则使用默认值（开发环境）
    demo_password = os.environ.get("DEMO_PASSWORD", "demopassword")
    demo_email = os.environ.get("DEMO_EMAIL", "demo@selfagi.com")
    demo_is_admin = os.environ.get("DEMO_IS_ADMIN", "false").lower() == "true"
    
    # 检查密码强度（最小长度8字符）
    if len(demo_password) < 8:
        print(f"警告：演示用户密码长度不足（{len(demo_password)}字符），建议至少8字符")
    
    db = SessionLocal()
    try:
        # 检查是否已存在演示用户
        existing_demo = db.query(User).filter(User.username == "demo").first()
        if existing_demo:
            print("演示用户已存在，更新密码和邮箱...")
            # 更新密码为新哈希
            existing_demo.hashed_password = get_password_hash(demo_password)
            existing_demo.email = demo_email
            existing_demo.is_admin = demo_is_admin
            existing_demo.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(existing_demo)
            print("演示用户信息已更新")
            return existing_demo
        
        # 创建新演示用户
        demo_user = User(
            username="demo",
            email=demo_email,
            hashed_password=get_password_hash(demo_password),
            full_name="演示用户",
            is_admin=demo_is_admin,  # 默认不是管理员，除非环境变量指定
            is_active=True,
            created_at=datetime.now(timezone.utc),
            last_login=None
        )
        
        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)
        
        print("演示用户创建成功")
        print(f"用户名: demo")
        print(f"邮箱: {demo_email}")
        print(f"管理员权限: {'是' if demo_is_admin else '否'}")
        print("提示：生产环境请通过环境变量DEMO_PASSWORD设置安全密码")
        
        return demo_user
    except Exception as e:
        print(f"创建演示用户失败: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("开始创建默认用户...")
    create_admin_user()
    create_demo_user()
    print("用户创建完成")