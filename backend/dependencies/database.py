"""
数据库依赖项
包含数据库会话依赖项
"""


from ..core.database import SessionLocal


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = ["get_db"]
