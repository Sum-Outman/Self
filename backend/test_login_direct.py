#!/usr/bin/env python3
"""直接测试登录功能，绕过FastAPI路由"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # backend的父目录
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db_models.user import User
from backend.core.config import Config
import hashlib
from passlib.context import CryptContext

# 使用与main.py一致的密码验证算法
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

def verify_password_local(plain_password: str, hashed_password: str) -> bool:
    """本地密码验证（与main.py一致）"""
    sha256_hash = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
    return pwd_context.verify(sha256_hash, hashed_password)

def test_database_connection():
    """测试数据库连接和用户查询"""
    print("=" * 60)
    print("直接数据库登录测试")
    print("=" * 60)
    
    # 创建数据库引擎
    database_url = Config.DATABASE_URL
    print(f"数据库URL: {database_url}")
    
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # 查询demo用户
        user = db.query(User).filter(User.username == "demo").first()
        
        if not user:
            print("❌ 错误：数据库中未找到demo用户")
            return False
        
        print(f"✅ 找到用户: {user.username}")
        print(f"   邮箱: {user.email}")
        print(f"   状态: {'激活' if user.is_active else '禁用'}")
        print(f"   密码哈希: {user.hashed_password[:30]}...")
        print(f"   创建时间: {user.created_at}")
        
        # 测试密码验证
        test_password = "demopassword"
        print(f"\n测试密码验证: '{test_password}'")
        
        # 使用本地验证
        result = verify_password_local(test_password, user.hashed_password)
        print(f"密码验证结果: {result}")
        
        if result:
            print("✅ 密码验证成功！")
        else:
            print("❌ 密码验证失败")
            
            # 尝试使用core.security验证（对比）
            from passlib.context import CryptContext
            pwd_context_pbkdf2 = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
            result_pbkdf2 = pwd_context_pbkdf2.verify(test_password, user.hashed_password)
            print(f"PBKDF2验证结果: {result_pbkdf2}")
            
            # 尝试原始SHA-256哈希
            sha256_hash = hashlib.sha256(test_password.encode("utf-8")).hexdigest()
            print(f"SHA-256哈希: {sha256_hash[:30]}...")
            
        return result
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

def test_auth_routes_login():
    """测试auth_routes.py中的登录函数"""
    print("\n" + "=" * 60)
    print("测试auth_routes.py登录函数")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        from backend.routes.auth_routes import verify_password as auth_verify_password
        from backend.routes.auth_routes import get_password_hash as auth_get_password_hash
        
        print("✅ 成功导入auth_routes.py中的验证函数")
        
        # 测试密码哈希生成
        test_password = "demopassword"
        test_hash = auth_get_password_hash(test_password)
        print(f"生成的哈希: {test_hash[:30]}...")
        
        # 测试验证
        result = auth_verify_password(test_password, test_hash)
        print(f"验证新哈希: {result}")
        
        # 从数据库获取实际哈希并测试
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.db_models.user import User
        
        engine = create_engine(Config.DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        user = db.query(User).filter(User.username == "demo").first()
        if user:
            print(f"\n数据库中的demo用户哈希: {user.hashed_password[:30]}...")
            result_db = auth_verify_password(test_password, user.hashed_password)
            print(f"验证数据库哈希: {result_db}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ 导入auth_routes.py失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fastapi_route():
    """测试FastAPI路由注册"""
    print("\n" + "=" * 60)
    print("测试FastAPI路由注册")
    print("=" * 60)
    
    try:
        # 检查路由是否在main.py中注册
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        if "auth_router" in content:
            print("✅ auth_router在main.py中被引用")
            
            # 检查导入语句
            import_lines = [line for line in content.split('\n') if "from backend.routes.auth_routes import" in line]
            if import_lines:
                print(f"✅ 导入语句: {import_lines[0]}")
            else:
                print("❌ 未找到auth_routes导入语句")
                
            # 检查include_router
            if "app.include_router(auth_router)" in content:
                print("✅ auth_router已注册到FastAPI应用")
            else:
                print("❌ auth_router未注册到FastAPI应用")
                
        else:
            print("❌ auth_router未在main.py中找到")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    print("Self AGI登录问题诊断")
    print("=" * 60)
    
    # 测试1: 直接数据库连接和验证
    db_result = test_database_connection()
    
    # 测试2: auth_routes函数导入
    auth_result = test_auth_routes_login()
    
    # 测试3: FastAPI路由注册
    test_fastapi_route()
    
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    if db_result:
        print("✅ 数据库连接和密码验证正常")
    else:
        print("❌ 数据库连接或密码验证失败")
        
    if auth_result:
        print("✅ auth_routes.py函数导入和调用正常")
    else:
        print("⚠️  auth_routes.py函数导入可能有问题")
        
    print("\n建议:")
    if not db_result:
        print("1. 检查数据库中的demo用户密码哈希")
        print("2. 运行create_admin.py重新创建demo用户")
        print("3. 验证密码哈希算法一致性")
    else:
        print("1. 数据库层面正常，问题可能在路由层")
        print("2. 检查FastAPI中间件是否拦截请求")
        print("3. 检查前端请求格式")