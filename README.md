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

## 📂 数据集准备

本项目严重依赖正确的数据路径和索引文件配置。

### 1\. 数据存放结构

智利矿井数据集应按 Session (采集架次) 文件夹存放。每个 Session 文件夹下**必须**包含点云文件夹和位置索引 CSV 文件。

推荐的目录结构如下：

```text
/path/to/dataset/
└── chilean_NoRot_NoScale/       # 数据集根目录 (RUNS_FOLDER)
    ├── 100/                     # Session ID
    │   ├── pointcloud_20m_10overlap/        # 存放 .bin 点云文件
    │   └── pointcloud_locations_20m_10overlap.csv  # 关键索引文件
    ├── 101/
    │   ├── ...
    └── ...
```

### 2\. 关键文件说明：pointcloud\_locations\_20m\_10overlap.csv

这是一个至关重要的索引文件，脚本会根据它来读取点云并确定其物理位置。**每个 Session 文件夹下都必须有这个文件。**

  * **作用**: 将点云文件名与物理坐标（Northing, Easting）关联，用于计算点云之间的距离，从而生成训练所需的正样本（Positives）和负样本（Negatives）。
  * **必需列 (Columns)**:
      * `timestamp`: 对应点云的文件名（不含后缀）。脚本会自动拼接为 `.bin` 文件路径。
      * `northing`: UTM 坐标 Y 轴。
      * `easting`: UTM 坐标 X 轴。
  * **使用方式**:
      * `generate_training_tuples_chilean.py` 和 `generate_test_sets_chilean.py` 脚本会读取该文件。
      * 脚本利用 `northing` 和 `easting` 构建 KDTree，以检索距离当前点云 7米以内（正样本）或 35米以外（负样本）的其他点云。

### 3\. 生成训练与测试索引

在使用模型前，必须先运行以下脚本生成 Pickle 格式的索引文件。

**步骤 A: 生成训练元组 (Training Tuples)**
运行脚本以划分训练集 (Session 100-159) 和测试集，并生成查询字典。

```bash
python datasets/chilean/generate_training_tuples_chilean.py
```

  * **输入**: 读取每个 Session 下的 `pointcloud_locations_20m_10overlap.csv`。
  * **输出**: `datasets/chilean/training_queries_chilean.pickle` (包含训练用的锚点、正样本、负样本索引)。

**步骤 B: 生成评估数据集 (Evaluation Sets)**
运行脚本以构建用于最终评估的 Database (历史地图, Session 160-189) 和 Query (当前观测, Session 190-209)。

```bash
python datasets/chilean/generate_test_sets_chilean.py
```

  * **输入**: 同样依赖 `pointcloud_locations_20m_10overlap.csv` 来确定 Database 和 Query 的真值位置。
  * **输出**: `datasets/chilean/chilean_evaluation_database_*.pickle` 和 `query_*.pickle`。

## 🚀 训练 (Training)

使用以下命令开始训练模型。

```bash
cd training
python train_chilean.py
```

  * **配置文件**: `config/config_chilean_baseline.txt`
      * 请务必修改配置文件中的 `dataset_folder` 为你的实际数据路径。
      * 默认 Batch Size: 128
      * Loss: TruncatedSmoothAP
  * **日志**: 训练日志将保存至 `training/trainer.log`。

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

## ⚙️ 关键配置修改指南

在运行代码前，请检查以下文件中的**绝对路径**设置：

1.  **`datasets/chilean/generate_training_tuples_chilean.py`** & **`generate_test_sets_chilean.py`**:

      * `BASE_PATH`: 修改为你的数据集根目录 (例如 `/data/Chilean_Dataset/`)。
      * `FILENAME`: 确认为 `"pointcloud_locations_20m_10overlap.csv"`。

2.  **`config/config_chilean_baseline.txt`**:

      * `dataset_folder`: 指向数据集根目录。
      * `train_file`: 指向生成的 `training_queries_chilean.pickle` 的绝对路径。

3.  **`datasets/pointnetvlad/pnv_raw.py`**:

      * `self.log_file`: 修改为你希望保存数据加载日志的路径。

## 📝 引用

本项目代码基于 [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2) 进行二次开发。
