# 循环导入问题修复指南

## 🔴 你遇到的错误

```
cannot import name 'MultiFlightGymEnv' from partially initialized module 'qln_multi_flight_game_v2' 
(most likely due to a circular import)
```

---

## 🎯 问题原因

### 循环导入是什么？

当两个 Python 模块相互导入对方时，就会发生循环导入。在你的情况下：

```
推理脚本名称: qln_multi_flight_game_v2_inference_xxx.py
环境模块名称: qln_multi_flight_game_v2.py
```

Python 可能混淆了这两个文件，因为：
1. **名字太相似** - 都以 `qln_multi_flight_game_v2` 开头
2. **在同一目录** - Python 导入时会优先查找当前目录
3. **推理脚本试图导入环境模块** - 但 Python 可能误认为是自己

### 具体场景

```python
# 在 qln_multi_flight_game_v2_inference_xxx.py 中
from qln_multi_flight_game_v2 import MultiFlightGymEnv

# Python 可能误解为：
# "哦，你要从 qln_multi_flight_game_v2_inference_xxx.py 中导入？"
# "但这个模块还没初始化完成！"
# → 循环导入错误！
```

---

## ✅ 解决方案

### 方案1：使用重命名的推理脚本（推荐）⭐

**使用新文件 `flight_inference.py`**：

```bash
# 复制到你的工作目录
cp flight_inference.py /Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/

# 运行
cd /Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/
python flight_inference.py

# 或者带参数
python flight_inference.py --flight1_days 4 --flight2_capacity 90
```

**为什么这样能解决？**
- ✅ 文件名 `flight_inference.py` 与 `qln_multi_flight_game_v2.py` 完全不同
- ✅ Python 不会混淆这两个模块
- ✅ 导入逻辑清晰明确

---

### 方案2：修改导入顺序

如果你想保留原文件名，可以调整导入顺序：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 关键：在导入环境之前先初始化 Ray
import ray
ray.init(ignore_reinit_error=True, log_to_driver=False)

# 然后才导入环境
from qln_multi_flight_game_v2 import MultiFlightGymEnv
```

---

### 方案3：使用绝对导入

```python
import sys
import os

# 确保 Python 知道从哪里导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 使用绝对导入
import qln_multi_flight_game_v2
from qln_multi_flight_game_v2 import MultiFlightGymEnv
```

---

### 方案4：延迟导入

```python
import ray
ray.init(...)

# 在需要的时候才导入
def get_env():
    from qln_multi_flight_game_v2 import MultiFlightGymEnv
    return MultiFlightGymEnv()

env = get_env()
```

---

## 📊 方案对比

| 方案 | 难度 | 可靠性 | 推荐度 |
|-----|------|--------|--------|
| 方案1: 重命名文件 | ⭐ 简单 | ⭐⭐⭐⭐⭐ 最可靠 | ⭐⭐⭐⭐⭐ 强烈推荐 |
| 方案2: 调整导入顺序 | ⭐⭐ 中等 | ⭐⭐⭐ 较可靠 | ⭐⭐⭐ 可以尝试 |
| 方案3: 绝对导入 | ⭐⭐⭐ 复杂 | ⭐⭐⭐⭐ 可靠 | ⭐⭐ 不太推荐 |
| 方案4: 延迟导入 | ⭐⭐ 中等 | ⭐⭐⭐ 较可靠 | ⭐⭐⭐ 可以尝试 |

---

## 🚀 快速开始（推荐方式）

### 步骤1：复制新文件

将 `flight_inference.py` 复制到你的工作目录：

```bash
# 假设你在 /workspace 目录
cp flight_inference.py /Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/
```

### 步骤2：进入工作目录

```bash
cd /Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/
```

### 步骤3：确认文件存在

```bash
ls -lh | grep -E "flight_inference|qln_multi_flight_game_v2"
```

你应该看到：
```
flight_inference.py           # 推理脚本（新）
qln_multi_flight_game_v2.py   # 环境定义
```

### 步骤4：运行

```bash
# 基本运行
python flight_inference.py

# 自定义参数（你的例子）
python flight_inference.py --flight1_days 4 --flight2_capacity 90

# 更多参数
python flight_inference.py \
    --episodes 5 \
    --flight1_days 25 \
    --flight1_capacity 150 \
    --flight2_days 28 \
    --flight2_capacity 200
```

---

## 🔍 验证环境文件

如果还是有问题，检查你的环境文件：

```bash
# 检查文件是否存在
ls -lh qln_multi_flight_game_v2.py

# 检查文件内容
head -20 qln_multi_flight_game_v2.py

# 尝试直接导入测试
python -c "from qln_multi_flight_game_v2 import MultiFlightGymEnv; print('导入成功')"
```

如果直接导入也失败，说明环境文件本身有问题：
1. 检查是否有语法错误
2. 检查是否缺少依赖
3. 检查类名是否正确

---

## 📝 文件结构建议

### 推荐的目录结构

```
nh_rms_pytorch/
├── qln_multi_flight_game_v2.py      # 环境定义
├── flight_inference.py              # 推理脚本（新名字）
├── train_ppo.py                     # 训练脚本
├── utils.py                         # 工具函数
└── README.md                        # 说明文档
```

### 避免的命名方式

❌ **不好**：
```
qln_multi_flight_game_v2.py              # 环境
qln_multi_flight_game_v2_inference.py    # 推理（太相似！）
qln_multi_flight_game_v2_train.py        # 训练（太相似！）
```

✅ **好**：
```
qln_multi_flight_game_v2.py    # 环境
flight_inference.py            # 推理（明确不同）
flight_train.py                # 训练（明确不同）
```

---

## 🛠️ 调试技巧

### 如果 flight_inference.py 还是报错

1. **检查 Python 路径**
```bash
python -c "import sys; print('\n'.join(sys.path))"
```

2. **检查是否有旧的 .pyc 文件**
```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

3. **使用 Python 调试模式**
```bash
python -v flight_inference.py 2>&1 | grep "qln_multi_flight"
```

4. **检查是否有同名文件夹**
```bash
find . -name "qln_multi_flight_game_v2*"
```

---

## 📚 扩展阅读

### Python 导入机制

Python 导入模块时的搜索顺序：
1. 当前目录
2. PYTHONPATH 环境变量
3. 标准库目录
4. 第三方包目录

### 避免循环导入的最佳实践

1. **清晰的命名** - 推理、训练、环境文件名要有明显区别
2. **模块化设计** - 环境定义独立，不依赖其他模块
3. **延迟导入** - 在函数内部导入，而不是文件顶部
4. **使用 `__name__ == '__main__'`** - 主执行逻辑放在这里

---

## ✨ 总结

### 立即解决方案

```bash
# 1. 停止旧进程
ray stop

# 2. 清理缓存
find . -name "*.pyc" -delete

# 3. 使用新文件
cd /Users/lifeng/Documents/ai_code/rms_pytorch/nh_rms_pytorch/
python flight_inference.py --flight1_days 4 --flight2_capacity 90
```

### 关键要点

1. **问题根源**：推理脚本和环境文件名太相似，导致 Python 混淆
2. **最佳方案**：使用 `flight_inference.py` 这个新名字
3. **为什么有效**：文件名完全不同，Python 不会混淆
4. **额外好处**：代码更清晰，维护更容易

### 预期输出

使用 `flight_inference.py` 后，你应该看到：

```
================================================================================
          多航班动态定价推理 - 修复版本
================================================================================

[1/5] 验证文件...
✓ Checkpoint 文件存在
✓ 准备导入环境...

[2/5] 初始化 Ray...
✓ Ray 2.x.x 初始化成功

[3/5] 导入和注册环境...
✓ 环境模块导入成功
✓ 环境 'multi_flight_v0' 注册成功

[4/5] 加载模型...
提示: 这一步可能需要30-90秒，请耐心等待...

[方法1] 使用轻量级配置加载...
  → 创建 Algorithm 对象...
  → 从 checkpoint 恢复权重...
  → 获取 policy...
✓ 方法1加载成功！

[5/5] 设置推理...
✓ 推理测试成功！示例动作: 2, 2

================================================================================
开始推理测试
================================================================================
...
```

如果你看到了 "✓ 环境模块导入成功"，说明循环导入问题已解决！
