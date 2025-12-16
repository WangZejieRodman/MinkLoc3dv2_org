# MinkLoc3Dv2 - Chilean Underground Mine Dataset

本项目基于 **MinkLoc3Dv2** 架构，针对 **智利地下矿井数据集 (Chilean Underground Mine Dataset)** 进行了适配、训练和评估。该项目旨在利用稀疏体素卷积网络（Sparse Voxel CNN）解决地下矿井环境中的点云位置识别（Place Recognition）和闭环检测问题。

## ✨ 主要特性

  * **核心模型**: 基于 MinkowskiEngine 的 MinkLoc3Dv2 (ResNet + FPN + ECA Block + GeM Pooling)。
  * **特定适配**: 针对地下矿井环境（隧道、无GPS）进行了数据加载和预处理适配（保留地面/顶板点云）。
  * **训练策略**: 支持 Truncated SmoothAP Loss，采用难样本挖掘（Hard Negative Mining）。
  * **评估体系**:
      * 标准的 Recall@N 和 Top 1% Recall 评估。
      * 跨 Session (时间段) 的训练与测试划分。
      * **旋转不变性测试**: 专门包含针对 Z 轴旋转鲁棒性的评估脚本。

## 🛠️ 环境依赖

请确保安装以下核心依赖库：

  * Python 3.x
  * PyTorch \>= 1.7
  * **MinkowskiEngine** (用于稀疏卷积)
  * NumPy, Pandas, Scipy, Sklearn
  * Open3D (可选，用于可视化)

## 📂 数据集准备

本项目严重依赖正确的数据路径配置。由于代码中包含硬编码路径（如 `/home/wzj/...`），**请务必在使用前修改相关路径**。

### 1\. 数据存放

请确保你的智利矿井数据集（`.bin` 格式点云）已按 Session 文件夹存放。

### 2\. 生成训练与测试索引

在使用模型前，需要生成用于检索的正负样本对索引文件（Pickle格式）。

**步骤 A: 生成训练元组 (Training Tuples)**
运行脚本以划分训练集 (Session 100-159) 和测试集，并生成查询字典。

```bash
python datasets/chilean/generate_training_tuples_chilean.py
```

  * **输出**: `training_queries_chilean.pickle`, `test_queries_chilean.pickle`
  * **注意**: 请检查脚本中的 `BASE_PATH` 和 `RUNS_FOLDER` 变量。

**步骤 B: 生成评估数据集 (Evaluation Sets)**
运行脚本以构建用于最终评估的 Database (历史地图, Session 160-189) 和 Query (当前观测, Session 190-209)。

```bash
python datasets/chilean/generate_test_sets_chilean.py
```

  * **输出**: `chilean_evaluation_database_*.pickle`, `chilean_evaluation_query_*.pickle`

## 🚀 训练 (Training)

使用以下命令开始训练模型。训练脚本会自动加载配置并进行模型优化。

```bash
cd training
python train_chilean.py
```

  * **配置文件**: `config/config_chilean_baseline.txt`
      * 默认 Batch Size: 128
      * Loss: TruncatedSmoothAP
      * 优化器: Adam
  * **模型结构**: 定义在 `models/minkloc3dv2.txt`
  * **日志**: 训练日志将保存至 `training/trainer.log`，权重保存至 `weights/` 目录。

## 📊 评估 (Evaluation)

### 1\. 标准评估

加载训练好的模型权重，并在测试集上计算 Recall@N。

```bash
cd eval
python evaluate_chilean.py
```

  * 该脚本会自动加载生成的 pickle 文件进行跨 Session 检索评估。
  * 需要修改脚本中的 `args.weights` 指向你训练好的 `.pth` 文件。

### 2\. 旋转不变性评估

评估模型在不同 Z 轴旋转角度（0°, 45°, 90°...）下的性能表现。

```bash
cd eval
python evaluate_chilean_rotation.py
```

  * 运行结束后，可使用 `python analyze_rotation_results.py` 生成详细的文本报告 (`rotation_results.txt`)。

## 📁 项目结构说明

```text
.
├── config/
│   └── config_chilean_baseline.txt    # 训练超参数配置文件
├── datasets/
│   ├── base_datasets.py               # 数据集基类
│   ├── augmentation.py                # 数据增强 (旋转, 翻转, 抖动)
│   ├── quantization.py                # 点云量化 (体素化)
│   ├── samplers.py                    # Batch Sampler (确保Batch内包含正样本对)
│   ├── chilean/                       # 智利数据集专用脚本
│   │   ├── generate_training_tuples_chilean.py
│   │   └── generate_test_sets_chilean.py
│   └── pointnetvlad/
│       ├── pnv_raw.py                 # 原始点云加载器 (不移除地面)
│       └── pnv_train.py               # 训练集特定Transform
├── models/
│   ├── minkloc.py                     # 模型主入口
│   ├── minkfpn.py                     # 特征金字塔网络 (Backbone)
│   ├── minkloc3dv2.txt                # 模型结构定义
│   └── layers/                        # 网络层 (Pooling, ECA Block, NetVLAD)
│   └── losses/                        # 损失函数 (TruncatedSmoothAP, Triplet)
├── training/
│   ├── train_chilean.py               # 训练启动脚本
│   └── trainer.py                     # 训练循环核心逻辑
├── eval/
│   ├── evaluate_chilean.py            # 标准评估脚本
│   ├── evaluate_chilean_rotation.py   # 旋转鲁棒性评估
│   └── analyze_rotation_results.py    # 旋转结果分析
└── misc/
    └── utils.py                       # 工具函数
```

## ⚙️ 关键配置修改指南

在运行代码前，请检查以下文件中的**绝对路径**设置：

1.  **`config/config_chilean_baseline.txt`**:

      * `dataset_folder`: 指向数据集根目录。
      * `train_file` / `val_file`: 指向生成的 pickle 文件路径。

2.  **`datasets/chilean/*.py`**:

      * `BASE_PATH`: 数据集存放位置。

3.  **`datasets/pointnetvlad/pnv_raw.py`**:

      * `self.log_file`: 日志输出路径。

4.  **`training/train_chilean.py`** 和 **`eval/*.py`**:

      * `args.weights`: 确保指向正确的模型权重文件。

## 📝 引用

本项目代码基于 [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2) 进行二次开发。

如果你在研究中使用了此代码，请引用原始 MinkLoc3D 论文以及相关数据集论文。
