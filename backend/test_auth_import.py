#!/usr/bin/env python3
"""测试auth_routes模块导入"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

print("测试auth_routes模块导入")
print("=" * 60)

try:
    # 尝试导入auth_routes模块
    print("1. 尝试导入auth_routes模块...")
    from backend.routes import auth_routes
    print(f"✅ 成功导入auth_routes模块: {auth_routes}")
    
    # 检查模块属性
    print("\n2. 检查模块属性...")
    print(f"   模块文件: {auth_routes.__file__}")
    
    # 检查router
    if hasattr(auth_routes, 'router'):
        print(f"✅ 找到router: {auth_routes.router}")
        print(f"   router前缀: {auth_routes.router.prefix}")
        print(f"   router标签: {auth_routes.router.tags}")
    else:
        print("❌ 未找到router属性")
    
    # 检查verify_password函数
    print("\n3. 检查verify_password函数...")
    if hasattr(auth_routes, 'verify_password'):
        print(f"✅ 找到verify_password函数: {auth_routes.verify_password}")
        
        # 测试函数
        test_password = "demopassword"
        test_hash = auth_routes.get_password_hash(test_password)
        print(f"   生成的测试哈希: {test_hash[:30]}...")
        
        result = auth_routes.verify_password(test_password, test_hash)
        print(f"   验证结果: {result}")
    else:
        print("❌ 未找到verify_password函数")
        
    # 检查login函数
    print("\n4. 检查login函数...")
    import inspect
    for name, obj in inspect.getmembers(auth_routes):
        if name == 'login' and inspect.isfunction(obj):
            print(f"✅ 找到login函数: {obj}")
            print(f"   函数签名: {inspect.signature(obj)}")
            break
    else:
        print("❌ 未找到login函数")
        
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ 其他错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("导入测试完成")