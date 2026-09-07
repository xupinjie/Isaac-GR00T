# HEVC 异步 GOP 视频训练复现

本文说明如何从原始数据开始，运行 200-step CPU 视频解码基线和异步 GOP 解码方案，并生成逐 step 性能对比结果。

## 1. 实验要求

### 1.1 软件与硬件

| 项目 | 要求 |
| --- | --- |
| 代码 | 包含本文档的 Isaac-GR00T commit |
| 容器镜像 | `nvcr.io/nvidia/pytorch:25.04-py3` |
| GPU | 单机 8 × NVIDIA A100 80GB |
| GPU 能力 | 容器内可使用 CUDA 和 NVIDIA 硬件视频解码 |
| 基础模型 | `nvidia/GR00T-N1.6-3B` |
| 数据集 | `unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset` |

使用其他型号或数量的 GPU 也可以验证功能，但不能直接与本文的性能数据比较。

### 1.2 固定训练配置

| 参数 | 值 |
| --- | ---: |
| GPU 数量 | 8 |
| Global batch size | 256 |
| 训练步数 | 200 |
| DataLoader workers / rank | 4 |
| Shard size | 512 |
| DataLoader 启动方式 | `fork` |
| Warmup ratio | 0.05 |

两个实验除视频后端外使用完全相同的配置：

- CPU 基线：TorchCodec CPU 解码；
- 优化方案：ACCV-Lab GPU 异步 GOP 解码。

## 2. 准备数据和模型

下载原始数据集：

```bash
hf download \
  unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset \
  --repo-type dataset \
  --local-dir /path/to/G1_Dex1_MountCameraRedGripper_Dataset
```

下载基础模型：

```bash
hf download \
  nvidia/GR00T-N1.6-3B \
  --local-dir /path/to/GR00T-N1.6-3B
```

如果数据集和模型已经存在，直接使用已有目录即可。

## 3. 启动容器

使用任意容器运行方式启动以下镜像，并将代码、数据集和模型挂载到容器内：

```text
nvcr.io/nvidia/pytorch:25.04-py3
```

以 Docker 为例：

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -v /path/to/workspace:/workspace \
  -w /workspace/Isaac-GR00T \
  nvcr.io/nvidia/pytorch:25.04-py3 \
  bash
```

其他容器系统只需提供等价的 GPU、视频解码和目录挂载能力。

## 4. 一键运行实验

进入容器后设置四个路径：

```bash
cd /workspace/Isaac-GR00T

export SOURCE_DATASET=/workspace/G1_Dex1_MountCameraRedGripper_Dataset
export HEVC_DATASET=/workspace/G1_Dex1_MountCameraRedGripper_Dataset_HEVC
export BASE_MODEL_PATH=/workspace/GR00T-N1.6-3B
export BENCHMARK_ROOT=/workspace/output_hevc_reproduction
```

执行入口脚本：

```bash
bash scripts/reproduce_hevc_training/run_reproduction.sh
```

入口脚本会自动完成：

1. 将数据集视频从 AV1 转换为 HEVC；
2. 准备 Python 环境和 ACCV-Lab decoder；
3. 运行 CPU TorchCodec 基线；
4. 运行异步 GOP 优化方案；
5. 汇总 200 个 step 的时间和 loss。

### HEVC 转换参数

转码使用 `libx265`、`medium` preset 和 CRF 23。仅改变视频编码格式，保持以下训练相关属性：

- 分辨率：640 × 480；
- 帧率：30 FPS；
- 像素格式：`yuv420p`；
- 帧数不变；
- GOP：2。

转换器会自动检查这些必要属性。已经正确转换的视频会被直接复用。

## 5. 输出结果

实验结束后，结果位于 `${BENCHMARK_ROOT}`：

```text
logs/
  cpu_torchcodec.log
  async_gop.log
raw/
  cpu_torchcodec/rank_00.json ... rank_07.json
  async_gop/rank_00.json ... rank_07.json
step_timing_comparison.csv
summary.json
reproduction_summary.json
comparison_table.md
```

每个 rank、每个 step 都记录：

- 等待数据时间；
- 训练时间；
- 总迭代时间；
- loss。

200-step 性能实验不保留 checkpoint。

## 6. 本次复现结果

测试环境为单机 8 × A100-SXM4-80GB，数据集视频为 HEVC、640×480、30 FPS、`yuv420p`、GOP=2。

| 指标 | CPU TorchCodec | 异步 GOP | 加速比 |
| --- | ---: | ---: | ---: |
| 200-step Trainer runtime | 672.47 s | **255.46 s** | **2.63×** |
| 训练吞吐 | 76.14 samples/s | **200.42 samples/s** | **2.63×** |
| 平均步时，排除前 10 步 | 2.5378 s | **1.1159 s** | **2.27×** |
| 数据等待，排除前 10 步 | 1.4387 s | **0.0753 s** | **19.11×** |
| Trainer train loss | 0.533712 | 0.483582 | — |

异步 GOP 方案将 200-step 训练时间降低 62.0%，吞吐提升 163.2%。两组 loss 均正常收敛；两条逐 step 平均 loss 曲线的 Pearson 相关系数为 0.9970。

不同机器的存储、CPU 和 GPU 拓扑会影响具体数值。复现时应重点比较同一台机器上两个视频后端的相对性能。
