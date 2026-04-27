---
title: PsiTrader C++ 量化交易系统
heroImage: /site-profile.png
---

# PsiTrader C++ 量化交易系统

PsiTrader 是一个高性能 C++ 量化交易系统，用于因子计算、回测和实盘交易。

## 项目结构

```
quant/psi-trader-zhangboxuan/
├── PsiFactor/           # 因子实现
│   ├── FactorBase.h    # 因子基类模板
│   ├── FactorType.hpp  # 因子类型枚举
│   └── *.cpp           # 具体因子实现
├── PsiTraderRunner/     # 回测运行器
│   ├── main.cpp        # 入口程序
│   └── config.yaml     # 配置文件
├── PsiData/            # 数据层
│   └── parquet/        # Parquet 读写
├── PsiCommon/          # 公共组件
├── PsiUtils/           # 工具函数
└── docs/              # 文档
```

## 编译指南

### 环境要求

- C++17 编译器 (g++/clang/MSVC)
- CMake 3.16+
- Eigen3 (可选，用于矩阵计算)
- Apache Arrow/Parquet (可选)

### 编译步骤

```bash
# 进入项目目录
cd quant/psi-trader-zhangboxuan

# 创建构建目录
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# 编译
cmake --build build --target PsiTraderRunner
```

## 因子开发流程

### 1. 新增因子类型

在 `PsiFactor/FactorType.hpp` 中添加枚举值：

```cpp
YourFactor = 5504, // 你的因子说明
```

### 2. 实现因子

参考 `PsiFactor/AvgVolCmp.h/cpp`：

```cpp
template<>
class FactorImpl<FactorType::YourFactor> : public FactorBase<YourFactorBar> {
public:
    explicit FactorImpl(const FactorParams& params);

    void computeTick(MarketDataField* data) override;
    void afterCompute() override;
    void reset() override;
};
```

### 3. 注册因子

在 `PsiFactor/FactorDispatch.cpp` 中 `#include` 新头文件。

### 4. 配置运行

修改 `PsiTraderRunner/config.yaml`：

```yaml
factors:
  - id: 5504
    params: [5, 10, 20]
```

### 5. 执行测试

```bash
# 运行回测
./build/PsiTraderRunner

# 检查输出 parquet
ls output/
```

## 详细文档

- [新因子开发与测试流程](/quant/新因子开发与测试流程) - 因子开发完整指南
- [个人量化系统搭建指南](/quant/个人量化系统搭建指南) - 量化学习路径

## 下一步

- 学习金融基础知识
- 研究因子有效性 (IC/IR)
- 搭建回测系统
- 准备模拟盘/实盘
