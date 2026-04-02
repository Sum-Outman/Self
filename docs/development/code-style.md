# Code Style Guide | 代码风格指南

This guide defines the coding standards and conventions for the Self AGI project. Consistent code style improves readability, maintainability, and collaboration.

本指南定义了 Self AGI 项目的编码标准和约定。一致的代码风格提高了可读性、可维护性和协作性。

## General Principles | 通用原则

### Readability | 可读性
- Code should be easy to read and understand
- Use meaningful names for variables, functions, and classes
- Keep functions small and focused on single responsibility
- Write self-documenting code with clear logic flow

- 代码应该易于阅读和理解
- 为变量、函数和类使用有意义的名称
- 保持函数小巧并专注于单一职责
- 编写具有清晰逻辑流的自文档化代码

### Maintainability | 可维护性
- Write code that is easy to modify and extend
- Avoid complex nested logic
- Use consistent patterns and conventions
- Document non-obvious logic and decisions

- 编写易于修改和扩展的代码
- 避免复杂的嵌套逻辑
- 使用一致的模式和约定
- 记录不明显的逻辑和决策

### Consistency | 一致性
- Follow established conventions throughout the codebase
- Use consistent formatting and naming
- Apply the same patterns for similar problems
- Update existing code to match new standards

- 在整个代码库中遵循已建立的约定
- 使用一致的格式和命名
- 对类似问题应用相同的模式
- 更新现有代码以匹配新标准

## Python Code Style | Python 代码风格

### PEP 8 Compliance | PEP 8 合规性
Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide with specific adaptations:

遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 风格指南并进行特定调整：

#### Indentation | 缩进
- Use 4 spaces per indentation level
- Never mix tabs and spaces
- Continuation lines should align with opening delimiter

- 每个缩进级别使用4个空格
- 切勿混合使用制表符和空格
- 续行应对齐开口分隔符

```python
# Good
def long_function_name(
        parameter_one, parameter_two,
        parameter_three, parameter_four):
    pass

# Bad - not aligned
def long_function_name(
    parameter_one, parameter_two,
    parameter_three, parameter_four):
    pass
```

#### Line Length | 行长度
- Maximum line length: 100 characters
- Use parentheses for line continuation
- Break lines before binary operators

- 最大行长度：100个字符
- 使用括号进行行续行
- 在二元运算符之前换行

```python
# Good
income = (gross_wages
          + taxable_interest
          + (dividends - qualified_dividends)
          - ira_deduction
          - student_loan_interest)

# Bad - line too long
income = gross_wages + taxable_interest + (dividends - qualified_dividends) - ira_deduction - student_loan_interest
```

#### Blank Lines | 空行
- Two blank lines between top-level functions and classes
- One blank line between method definitions
- Use blank lines sparingly within functions

- 顶级函数和类之间使用两个空行
- 方法定义之间使用一个空行
- 在函数内谨慎使用空行

```python
import os
import sys


class MyClass:
    """Class documentation."""
    
    def method_one(self):
        pass
    
    def method_two(self):
        pass


def top_level_function():
    pass


def another_function():
    pass
```

#### Imports | 导入
- Group imports in this order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library imports
- Use absolute imports
- Avoid wildcard imports (`from module import *`)

- 按此顺序分组导入：
  1. 标准库导入
  2. 相关的第三方导入
  3. 本地应用程序/库导入
- 使用绝对导入
- 避免通配符导入（`from module import *`）

```python
# Good
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.models.user import User
from app.services.auth import authenticate

# Bad - mixed order
from app.models.user import User
import os
import numpy as np
from typing import Dict
```

### Naming Conventions | 命名约定

#### Variables and Functions | 变量和函数
- Use `snake_case` for variable and function names
- Use descriptive names that indicate purpose
- Avoid single-letter names except for iterators

- 变量和函数名使用 `snake_case`
- 使用表明用途的描述性名称
- 除了迭代器外，避免使用单字母名称

```python
# Good
user_count = 0
max_retry_attempts = 3

def calculate_total_price(items):
    pass

def validate_user_input(input_data):
    pass

# Bad
cnt = 0
mra = 3

def calc(items):
    pass

def val(input):
    pass
```

#### Classes | 类
- Use `PascalCase` for class names
- Use nouns or noun phrases
- Avoid abbreviations

- 类名使用 `PascalCase`
- 使用名词或名词短语
- 避免缩写

```python
# Good
class UserManager:
    pass

class DataProcessor:
    pass

class NeuralNetworkModel:
    pass

# Bad
class user_manager:
    pass

class DP:
    pass

class neuralnetworkmodel:
    pass
```

#### Constants | 常量
- Use `UPPER_CASE_WITH_UNDERSCORES`
- Define at module level

- 使用 `UPPER_CASE_WITH_UNDERSCORES`
- 在模块级别定义

```python
# Good
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT = 30
API_VERSION = "v1.0"

# Bad
maxConnections = 100
default_timeout = 30
ApiVersion = "v1.0"
```

#### Private Members | 私有成员
- Use single leading underscore for non-public methods and instance variables
- Use double leading underscore for name mangling (rarely needed)

- 非公共方法和实例变量使用单个前导下划线
- 名称修饰使用双前导下划线（很少需要）

```python
class MyClass:
    def __init__(self):
        self.public_var = "public"
        self._protected_var = "protected"
        self.__private_var = "private"
    
    def public_method(self):
        pass
    
    def _protected_method(self):
        pass
    
    def __private_method(self):
        pass
```

### Type Annotations | 类型注解
- Use type hints for all function parameters and return values
- Use `Optional` for parameters that can be `None`
- Use `Union` for multiple possible types
- Use `Any` sparingly

- 所有函数参数和返回值都使用类型提示
- 可以为 `None` 的参数使用 `Optional`
- 多种可能类型使用 `Union`
- 谨慎使用 `Any`

```python
from typing import List, Dict, Optional, Union, Any

def process_data(
    data: List[Dict[str, Any]],
    config: Optional[Dict[str, Union[str, int, float]]] = None
) -> Dict[str, Any]:
    """Process data with optional configuration."""
    if config is None:
        config = {}
    
    result = {}
    # Processing logic
    return result

class User:
    def __init__(self, name: str, age: int, email: Optional[str] = None):
        self.name = name
        self.age = age
        self.email = email
    
    def get_info(self) -> Dict[str, Union[str, int]]:
        return {
            "name": self.name,
            "age": self.age,
            "email": self.email or "No email"
        }
```

### Documentation Strings | 文档字符串

#### Function Docstrings | 函数文档字符串
- Use triple double quotes (`"""`)
- Follow Google style docstring format
- Include: description, args, returns, raises, examples

- 使用三重双引号（`"""`）
- 遵循 Google 风格的文档字符串格式
- 包括：描述、参数、返回值、异常、示例

```python
def calculate_statistics(
    data: List[float],
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate descriptive statistics for numerical data.
    
    Args:
        data: List of numerical values to analyze
        weights: Optional list of weights for weighted statistics
    
    Returns:
        Dictionary containing mean, median, std, min, and max
    
    Raises:
        ValueError: If data is empty or contains non-numeric values
        TypeError: If weights length doesn't match data length
    
    Examples:
        >>> data = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> calculate_statistics(data)
        {'mean': 3.0, 'median': 3.0, 'std': 1.58, 'min': 1.0, 'max': 5.0}
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Implementation
    result = {
        "mean": sum(data) / len(data),
        # ... other calculations
    }
    return result
```

#### Class Docstrings | 类文档字符串
- Include class purpose and usage
- Document public attributes and methods
- Include examples if helpful

- 包括类的用途和用法
- 记录公共属性和方法
- 如果有帮助，包括示例

```python
class DataTransformer:
    """
    Transform data between different formats and structures.
    
    This class provides methods for converting data between various
    formats including JSON, CSV, and pandas DataFrames.
    
    Attributes:
        source_format (str): Format of input data
        target_format (str): Desired output format
        options (Dict): Transformation options
    
    Examples:
        >>> transformer = DataTransformer("json", "csv")
        >>> result = transformer.transform('{"id": 1, "value": "test"}')
        'id,value\\n1,test'
    """
    
    def __init__(self, source_format: str, target_format: str, **options):
        """Initialize transformer with source and target formats."""
        self.source_format = source_format
        self.target_format = target_format
        self.options = options
    
    def transform(self, data: str) -> str:
        """Transform data from source to target format."""
        # Implementation
        pass
```

#### Module Docstrings | 模块文档字符串
- Place at top of file
- Include module purpose and key exports
- Document important functions and classes

- 放在文件顶部
- 包括模块用途和关键导出
- 记录重要函数和类

```python
"""
Data processing utilities for the Self AGI system.

This module provides functions and classes for processing various
types of data including text, images, and multimodal data.

Key Functions:
    - preprocess_text: Clean and normalize text data
    - extract_features: Extract features from raw data
    - validate_data: Validate data quality and consistency

Key Classes:
    - DataProcessor: Main data processing pipeline
    - FeatureExtractor: Extract features from different data types
    - DataValidator: Validate data quality

Example:
    >>> from app.data import preprocess_text
    >>> cleaned = preprocess_text("Raw text with   extra spaces.")
    'Raw text with extra spaces.'
"""

import re
from typing import List, Dict, Any

# Module implementation...
```

### Error Handling | 错误处理

#### Exception Types | 异常类型
- Use built-in exceptions when appropriate
- Create custom exceptions for domain-specific errors
- Make exceptions informative with clear messages

- 适当时使用内置异常
- 为特定领域错误创建自定义异常
- 使用清晰的消息使异常信息丰富

```python
# Custom exceptions
class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass

class ResourceExhaustedError(Exception):
    """Raised when system resources are exhausted."""
    pass

# Usage
def validate_data(data: Dict[str, Any]) -> bool:
    """Validate input data."""
    if not data:
        raise DataValidationError("Data cannot be empty")
    
    if "required_field" not in data:
        raise DataValidationError("Missing required field: required_field")
    
    # Additional validation
    return True
```

#### Try-Except Blocks | Try-Except 块
- Be specific about exceptions to catch
- Handle exceptions at appropriate levels
- Log exceptions before re-raising if needed
- Clean up resources in finally blocks

- 明确要捕获的异常
- 在适当的级别处理异常
- 如果需要重新抛出异常，先记录异常
- 在 finally 块中清理资源

```python
import logging

logger = logging.getLogger(__name__)

def process_file(file_path: str) -> Optional[Dict]:
    """Process file with proper error handling."""
    file = None
    try:
        file = open(file_path, 'r')
        data = json.load(file)
        
        # Validate data
        if not validate_data(data):
            raise DataValidationError("Invalid data format")
        
        # Process data
        result = process_data(data)
        return result
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise DataValidationError("Invalid JSON format") from e
        
    except DataValidationError as e:
        logger.warning(f"Data validation failed: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error processing file {file_path}: {e}")
        raise
        
    finally:
        if file:
            file.close()
```

### Testing Style | 测试风格

#### Test Naming | 测试命名
- Use descriptive test names
- Prefix with `test_`
- Indicate what is being tested and expected outcome

- 使用描述性的测试名称
- 以 `test_` 为前缀
- 指示正在测试的内容和预期结果

```python
# Good
def test_user_creation_with_valid_data():
    pass

def test_login_fails_with_invalid_credentials():
    pass

def test_data_processing_handles_empty_input():
    pass

# Bad
def test1():
    pass

def test_user():
    pass

def test_thing():
    pass
```

#### Test Structure | 测试结构
- Use Arrange-Act-Assert pattern
- Keep tests independent
- Use fixtures for setup and teardown

- 使用 Arrange-Act-Assert 模式
- 保持测试独立
- 使用 fixture 进行设置和清理

```python
import pytest
from app.models.user import User
from app.services.auth import authenticate

def test_authenticate_with_valid_credentials():
    """Test authentication with valid credentials."""
    # Arrange
    username = "testuser"
    password = "password123"
    user = User.create(username=username, password=password)
    
    # Act
    result = authenticate(username=username, password=password)
    
    # Assert
    assert result is True
    assert user.is_authenticated

def test_authenticate_with_invalid_password():
    """Test authentication fails with invalid password."""
    # Arrange
    username = "testuser"
    user = User.create(username=username, password="correct")
    
    # Act
    result = authenticate(username=username, password="wrong")
    
    # Assert
    assert result is False
    assert not user.is_authenticated

@pytest.fixture
def test_user():
    """Fixture for creating test user."""
    user = User.create(username="fixture_user", password="test")
    yield user
    user.delete()

def test_user_with_fixture(test_user):
    """Test using fixture."""
    assert test_user.username == "fixture_user"
    assert test_user.is_active
```

## JavaScript/TypeScript Code Style | JavaScript/TypeScript 代码风格

### General Conventions | 通用约定

#### Naming | 命名
- Use `camelCase` for variables and functions
- Use `PascalCase` for classes and interfaces
- Use `UPPER_CASE` for constants
- Use meaningful, descriptive names

- 变量和函数使用 `camelCase`
- 类和接口使用 `PascalCase`
- 常量使用 `UPPER_CASE`
- 使用有意义、描述性的名称

```typescript
// Good
const userCount = 0;
const MAX_RETRIES = 3;

function calculateTotalPrice(items: Item[]): number {
  // ...
}

class UserManager {
  // ...
}

interface UserData {
  id: number;
  name: string;
}

// Bad
const usercount = 0;
const MaxRetries = 3;

function calc(items: Item[]): number {
  // ...
}

class user_manager {
  // ...
}
```

#### Type Annotations | 类型注解
- Use TypeScript for all new code
- Provide explicit types for function parameters and return values
- Use interfaces for object shapes
- Avoid `any` type when possible

- 所有新代码都使用 TypeScript
- 为函数参数和返回值提供显式类型
- 使用接口定义对象形状
- 尽可能避免使用 `any` 类型

```typescript
interface User {
  id: number;
  name: string;
  email?: string;  // Optional
}

interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}

function getUserById(id: number): Promise<ApiResponse<User>> {
  // ...
}

function processItems(items: Item[], options?: ProcessOptions): ProcessResult {
  // ...
}
```

### React/Component Style | React/组件风格

#### Functional Components | 函数组件
- Use functional components with hooks
- Use destructuring for props
- Keep components small and focused
- Use TypeScript for props

- 使用带有钩子的函数组件
- 对 props 使用解构
- 保持组件小巧且专注
- 对 props 使用 TypeScript

```typescript
import React, { useState, useEffect } from 'react';

interface UserCardProps {
  user: User;
  onEdit?: (user: User) => void;
  onDelete?: (userId: number) => void;
  className?: string;
}

const UserCard: React.FC<UserCardProps> = ({
  user,
  onEdit,
  onDelete,
  className = ''
}) => {
  const [isEditing, setIsEditing] = useState(false);
  
  useEffect(() => {
    // Component did mount/update logic
  }, [user]);
  
  const handleEdit = () => {
    setIsEditing(true);
    onEdit?.(user);
  };
  
  const handleDelete = () => {
    if (window.confirm('Are you sure?')) {
      onDelete?.(user.id);
    }
  };
  
  return (
    <div className={`user-card ${className}`}>
      <h3>{user.name}</h3>
      <p>Email: {user.email}</p>
      <div className="actions">
        <button onClick={handleEdit}>Edit</button>
        <button onClick={handleDelete}>Delete</button>
      </div>
    </div>
  );
};

export default UserCard;
```

#### Hooks | 钩子
- Use custom hooks for reusable logic
- Follow rules of hooks
- Name hooks with `use` prefix
- Keep hooks focused on single concern

- 使用自定义钩子实现可重用逻辑
- 遵循钩子的规则
- 钩子名称以 `use` 为前缀
- 保持钩子专注于单一关注点

```typescript
import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

interface UseFetchOptions {
  initialData?: any;
  autoFetch?: boolean;
}

function useFetch<T>(url: string, options: UseFetchOptions = {}) {
  const { initialData = null, autoFetch = true } = options;
  
  const [data, setData] = useState<T | null>(initialData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.get<T>(url);
      setData(response.data);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, [url]);
  
  useEffect(() => {
    if (autoFetch) {
      fetchData();
    }
  }, [fetchData, autoFetch]);
  
  return {
    data,
    loading,
    error,
    refetch: fetchData,
    setData
  };
}

// Usage
const UserList: React.FC = () => {
  const { data: users, loading, error, refetch } = useFetch<User[]>('/api/users');
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  
  return (
    <div>
      <h2>Users</h2>
      <button onClick={refetch}>Refresh</button>
      {/* Render users */}
    </div>
  );
};
```

### Formatting | 格式化

#### Indentation and Spacing | 缩进和间距
- Use 2 spaces for indentation
- Use semicolons consistently
- Add spaces around operators
- Use trailing commas in multiline objects/arrays

- 缩进使用2个空格
- 一致使用分号
- 在运算符周围添加空格
- 在多行对象/数组中使用尾随逗号

```typescript
// Good
const user = {
  id: 1,
  name: 'John Doe',
  email: 'john@example.com',
};

const numbers = [
  1,
  2,
  3,
  4,
  5,
];

function calculate(a: number, b: number): number {
  const sum = a + b;
  const product = a * b;
  return sum + product;
}

// Bad
const user = {id:1,name:'John Doe',email:'john@example.com'};

function calculate(a:number,b:number):number{
const sum=a+b
const product=a*b
return sum+product}
```

#### Line Length | 行长度
- Maximum line length: 100 characters
- Break long lines at logical points
- Use template literals for long strings

- 最大行长度：100个字符
- 在逻辑点处换行
- 长字符串使用模板字面量

```typescript
// Good
const errorMessage = `Failed to process request. Please check your input and try again. ` +
  `If the problem persists, contact support at support@example.com.`;

const longUrl = 'https://api.example.com/v1/users' +
  `?filter=active` +
  `&sort=name` +
  `&page=${page}` +
  `&limit=${limit}`;

// Bad
const errorMessage = 'Failed to process request. Please check your input and try again. If the problem persists, contact support at support@example.com.';
```

## Database Code Style | 数据库代码风格

### SQL Formatting | SQL 格式化
- Use uppercase for SQL keywords
- Indent subqueries and JOIN clauses
- Align columns in SELECT statements
- Use meaningful table and column names

- SQL 关键字使用大写
- 缩进子查询和 JOIN 子句
- 在 SELECT 语句中对齐列
- 使用有意义的表和列名

```sql
-- Good
SELECT
    u.id AS user_id,
    u.username,
    u.email,
    COUNT(o.id) AS order_count,
    SUM(o.total) AS total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.active = TRUE
    AND u.created_at >= '2024-01-01'
GROUP BY u.id, u.username, u.email
HAVING COUNT(o.id) > 0
ORDER BY total_spent DESC
LIMIT 100;

-- Bad
select u.id as user_id, u.username, u.email, count(o.id) as order_count, sum(o.total) as total_spent from users u left join orders o on u.id = o.user_id where u.active = true and u.created_at >= '2024-01-01' group by u.id, u.username, u.email having count(o.id) > 0 order by total_spent desc limit 100;
```

### ORM Code Style | ORM 代码风格

#### SQLAlchemy (Python) | SQLAlchemy (Python)
- Use declarative base for models
- Define relationships clearly
- Use type hints for model attributes
- Follow naming conventions

- 模型使用声明式基类
- 明确定义关系
- 模型属性使用类型提示
- 遵循命名约定

```python
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from typing import Optional
import datetime

Base = declarative_base()

class User(Base):
    """User model representing system users."""
    
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="user")
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"

class Post(Base):
    """Blog post model."""
    
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    author = relationship("User", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Post(id={self.id}, title='{self.title}')>"
```

## Configuration and Environment | 配置和环境

### Environment Variables | 环境变量
- Use uppercase with underscores for environment variable names
- Provide defaults where appropriate
- Validate required environment variables at startup
- Document all environment variables

- 环境变量名使用大写加下划线
- 适当时提供默认值
- 启动时验证必需的环境变量
- 记录所有环境变量

```python
import os
from typing import Optional
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Required variables
    DATABASE_URL: str
    SECRET_KEY: str
    
    # Optional variables with defaults
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_PORT: int = 8000
    REDIS_URL: Optional[str] = None
    
    # Validation
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v.startswith(("postgresql://", "sqlite://")):
            raise ValueError("Invalid database URL format")
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Usage
settings = Settings()
```

### Configuration Files | 配置文件
- Use YAML or JSON for configuration files
- Use consistent structure across environments
- Validate configuration on load
- Document all configuration options

- 配置文件使用 YAML 或 JSON
- 跨环境使用一致的结构
- 加载时验证配置
- 记录所有配置选项

```yaml
# config.yaml
app:
  name: "Self AGI"
  version: "1.0.0"
  debug: false
  port: 8000

database:
  url: "postgresql://user:password@localhost:5432/self_agi"
  pool_size: 20
  max_overflow: 40
  echo: false

model:
  name: "self_agi_v1"
  parameters:
    hidden_size: 768
    num_layers: 12
    num_heads: 12
    dropout: 0.1
  
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  checkpoint_dir: "./checkpoints"
```

## File Organization | 文件组织

### Module Structure | 模块结构
- Keep related functionality together
- Limit module size (aim for < 500 lines)
- Use `__init__.py` to expose public API
- Group by feature or layer, not by type

- 将相关功能放在一起
- 限制模块大小（目标 < 500 行）
- 使用 `__init__.py` 暴露公共 API
- 按功能或层分组，而不是按类型

```
# Good structure
app/
├── users/
│   ├── __init__.py      # Exports: User, UserService, UserRepository
│   ├── models.py        # User model classes
│   ├── services.py      # User business logic
│   ├── repositories.py  # User data access
│   └── schemas.py       # User Pydantic schemas
├── posts/
│   ├── __init__.py
│   ├── models.py
│   ├── services.py
│   └── schemas.py
└── utils/
    ├── __init__.py
    ├── validation.py
    └── formatting.py

# Bad structure - grouped by type
app/
├── models/
│   ├── user.py
│   ├── post.py
│   └── comment.py
├── services/
│   ├── user_service.py
│   ├── post_service.py
│   └── comment_service.py
└── repositories/
    ├── user_repo.py
    ├── post_repo.py
    └── comment_repo.py
```

### Import Statements | 导入语句
- Use relative imports within the same package
- Use absolute imports for cross-package imports
- Avoid circular imports
- Keep imports organized

- 同一包内使用相对导入
- 跨包导入使用绝对导入
- 避免循环导入
- 保持导入有序

```python
# Good - within app.users package
from .models import User
from .schemas import UserCreate, UserUpdate
from ..utils.validation import validate_email

# Good - from another package
from app.users.models import User
from app.posts.services import PostService

# Bad - mixing styles
from models import User
from app.utils.validation import validate_email
import services.user_service
```

## Tooling and Automation | 工具和自动化

### Linting | 代码检查
- Use flake8 for Python linting
- Use ESLint for JavaScript/TypeScript
- Configure pre-commit hooks
- Fix linting errors before committing

- Python 代码检查使用 flake8
- JavaScript/TypeScript 使用 ESLint
- 配置预提交钩子
- 提交前修复代码检查错误

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
```

### Formatting | 格式化
- Use black for Python formatting
- Use Prettier for frontend code
- Configure IDE to format on save
- Ensure consistent formatting across team

- Python 格式化使用 black
- 前端代码使用 Prettier
- 配置 IDE 在保存时格式化
- 确保团队间一致的格式化

```json
// .prettierrc.json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false
}
```

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
```

## Code Review Checklist | 代码审查清单

### Before Submission | 提交前
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No linting errors
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed
- [ ] Error handling is adequate

- [ ] 代码遵循风格指南
- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] 没有代码检查错误
- [ ] 没有安全漏洞
- [ ] 性能考虑已解决
- [ ] 错误处理足够

### During Review | 审查期间
- [ ] Code is readable and maintainable
- [ ] Logic is correct and efficient
- [ ] Edge cases are handled
- [ ] Tests cover functionality
- [ ] Security best practices followed
- [ ] No unnecessary complexity
- [ ] Follows established patterns

- [ ] 代码可读且可维护
- [ ] 逻辑正确且高效
- [ ] 处理了边缘情况
- [ ] 测试覆盖功能
- [ ] 遵循安全最佳实践
- [ ] 没有不必要的复杂性
- [ ] 遵循已建立的模式

### After Review | 审查后
- [ ] All feedback addressed
- [ ] Changes tested locally
- [ ] Documentation updated if needed
- [ ] Ready for merge

- [ ] 所有反馈已解决
- [ ] 更改已在本地测试
- [ ] 如果需要，文档已更新
- [ ] 准备合并

---

*Last Updated: March 30, 2026*  
*最后更新: 2026年3月30日*