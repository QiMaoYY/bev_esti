# bev_esti

`bev_esti` 是一个独立的、本地 CPU 友好的 BEV 位姿估计小仓库。

它的目标不是重新训练 `BEVPlace2`，而是复用已经训练好的模型，在本地完成：

1. 单张查询 BEV 图像的特征提取
2. 与离线数据库 BEV 的粗检索
3. 参考 `BEVPlace2` 后半段源码完成局部特征匹配与 `RANSAC` 位姿恢复
4. 输出估计的 3DoF 位姿 `(x, y, yaw)`

---

## 1. 输入与输出

### 输入

- 训练好的模型 checkpoint
- 离线数据库样本表 `database_samples.csv`
- 当前视角的单张查询 BEV 图像
- 可选：预先缓存好的数据库全局描述子 `npz`

### 输出

- 数据库 Top-K 粗匹配候选
- 最优候选对应的估计位姿
- 若提供查询集真实值，还可同时输出 `xy` 和 `yaw` 误差

---

## 2. 仓库结构

```text
bev_esti/
  README.md
  requirements.txt
  setup_cpu_env.sh
  build_db_cache.py
  estimate_pose.py
  batch_evaluate.py
  src/
    __init__.py
    model.py
    data.py
    ransac.py
    runtime.py
    visualization.py
    project_paths.py
```

说明：

- `model.py`：独立封装 `REIN / REM / NetVLAD`
- `data.py`：读取 `database_samples.csv / query_samples.csv` 和 BEV 图像
- `ransac.py`：移植 `BEVPlace2/RANSAC.py`
- `runtime.py`：数据库描述子提取、Top-K 搜索、局部特征匹配、位姿恢复
- `visualization.py`：单样本 / 批量可视化导出
- `project_paths.py`：默认路径、自动 checkpoint 检测、自动输出目录命名
- `build_db_cache.py`：离线缓存数据库全局描述子
- `estimate_pose.py`：对单个 query BEV 估计位姿
- `batch_evaluate.py`：批量评测“检索锚点位姿 / BEVPlace++ 3DoF 粗定位结果”并可选导出可视化

---

## 3. 为什么默认不使用 conda

当前本机已有：

- `python3`
- `venv`
- `torch`
- `opencv`
- `numpy`

并且当前 `bev_esti` 已经内置了 `ResNet34` 主干实现，因此：

- **不再依赖 `torchvision`**
- **不依赖 `faiss`**

其中：

- `faiss` 对这个仓库不是必须的，因为这里直接用 `numpy` 做数据库描述子 L2 搜索

因此本仓库优先采用：

> `venv + pip`

的方式构建本地 CPU 推理环境，不默认依赖 `conda`。

只有在后续遇到：

- `torch/torchvision` 与本机 Python 版本无法兼容
- 或本机环境污染严重无法隔离

时，才建议再考虑 `miniconda`。

---

## 4. CPU 环境安装

### 4.1 推荐方式：venv

在仓库根目录执行：

```bash
cd /home/qimao/grad_ws/bev_esti
bash ./setup_cpu_env.sh
source ./.venv/bin/activate
```

这个脚本会执行：

1. 创建本地虚拟环境 `.venv`
2. 优先安装你本地已经下载好的 `torch` wheel
3. 通过清华源安装：
   - `numpy<2`
   - `opencv-python==4.11.0.86`
   - `tqdm`

如果当前本机系统 Python 已经能直接导入：

- `torch`
- `cv2`
- `numpy`

则也可以**不进 venv，直接用系统 Python 运行**。  
本次本机实际验证时就是采用这一方式完成的。

### 4.2 为什么这里不用 `torchvision` 和 `faiss`

当前 `bev_esti` 中已经内置：

- `ResNet34` 结构定义

因此不再依赖 `torchvision.models.resnet34`。

当前数据库规模只有几百条，单次查询的 CPU 检索完全可以用：

- `numpy` 直接计算描述子 L2 距离

这样做的好处是：

- 少一个依赖
- 避免本机额外安装 `faiss-cpu`
- 避免为了单机推理专门解决 `torchvision` 安装兼容问题
- 更适合轻量推理仓库

---

## 5. 与当前项目数据的对接方式

当前默认使用以下数据表：

- 数据库样本表：
  - `/home/qimao/grad_ws/data/bevplace_tables/database_samples.csv`
- 查询样本表：
  - `/home/qimao/grad_ws/data/bevplace_tables/query_samples.csv`

默认数据根目录：

- `/home/qimao/grad_ws/data`

默认 checkpoint：

- 优先读取环境变量 `BEV_ESTI_CHECKPOINT`
- 若 `db-cache` 中已经记录了构建它所用的 checkpoint，则默认直接复用该 checkpoint
- 若既没有环境变量也没有缓存元信息，则自动选择 `../BEVPlace2/runs/` 下最近的 `bevplace` 训练 checkpoint

默认输出目录：

- 单样本 JSON：`debug_outputs/estimate_pose/<query_key>/result.json`
- 单样本可视化：`debug_outputs/estimate_pose/<query_key>/viz/`
- 批量评测：`debug_outputs/batch_evaluate/range_xxxx_yyyy/`

数据库样本表中保存了：

- 数据库 BEV 图像路径
- 数据库点云路径
- 数据库锚点位姿 `anchor_x / anchor_y / anchor_yaw_rad`

因此本仓库可以直接把数据库 BEV 与数据库位姿联系起来，进而把相对位姿恢复结果转换为全局位姿。

---

## 6. 先离线建立数据库缓存

如果每次估计前都重新提取整个数据库的全局描述子，会比较慢。  
因此建议先构建数据库缓存：

```bash
cd /home/qimao/grad_ws/bev_esti
python3 build_db_cache.py \
  --device cpu \
  --batch-size 8 \
  --num-workers 0
```

说明：

- 新版 `build_db_cache.py` 会把 `checkpoint_path` 一起写进 `database_cache.npz`
- 如果你手头已有旧版 `database_cache.npz`，它不带这个元信息
- 这种旧缓存现在不会再被脚本静默拿来和别的 checkpoint 混用，而是会提示你显式传 `--checkpoint` 或先重建缓存

如果想手动指定模型，也可以额外传入：

- `--checkpoint /path/to/model_best.pth.tar`

输出：

- `database_cache.npz`

其中包含：

- 数据库全局描述子
- 数据库样本索引
- 数据库样本关键字段快照

---

## 7. 对单张查询 BEV 估计位姿

### 7.1 直接输入一张 BEV 图片

```bash
cd /home/qimao/grad_ws/bev_esti
python3 estimate_pose.py \
  --query-image /home/qimao/grad_ws/data/icp_segments_noground/bev_single/seg_0001.png \
  --topk 5 \
  --device cpu
```

默认会输出到：

- `debug_outputs/estimate_pose/seg_0001/result.json`

### 7.2 使用查询样本表中的某个 query 做评测

```bash
cd /home/qimao/grad_ws/bev_esti
python3 estimate_pose.py \
  --query-index 10 \
  --topk 5 \
  --device cpu
```

这时脚本会同时输出：

- 估计位姿
- 查询真实值
- 估计误差
- 对应 JSON 默认保存到 `debug_outputs/estimate_pose/query_0010/result.json`

### 7.3 导出 Top-K 变换关系与局部特征可视化

如果希望进一步检查：

- 输入 query BEV 与 Top-K 候选 BEV 的对齐关系
- `FAST` 关键点分布
- 局部特征的伪彩色可视化
- 局部匹配与 `RANSAC` 内点情况

可以在单样本推理时附加 `--visualize`：

```bash
cd /home/qimao/grad_ws/bev_esti
python3 estimate_pose.py \
  --query-index 10 \
  --topk 3 \
  --device cpu \
  --visualize
```

默认会导出到：

- `debug_outputs/estimate_pose/query_0010/result.json`
- `debug_outputs/estimate_pose/query_0010/viz/topk_summary.png`

默认会导出：

- `topk_summary.png`：Top-K 候选总览
- `rank_XX_<db_key>_panel.png`：单个候选的详细可视化面板

其中每个候选面板包含：

- query / candidate 原始 BEV
- 变换前后叠加图
- query / candidate 的 `FAST` 关键点
- query / candidate 的局部特征伪彩色图
- 局部匹配连线图，其中绿色表示 `RANSAC` 内点

如果需要减少匹配连线数量或强制指定目录，可额外设置：

- `--visualize-match-limit`
- `--visualize-dir`

### 7.4 批量评测前 20 个查询样本

```bash
cd /home/qimao/grad_ws/bev_esti
python3 batch_evaluate.py \
  --start-index 0 \
  --limit 20 \
  --topk 5 \
  --device cpu
```

输出：

- `debug_outputs/batch_evaluate/range_0000_0019/results.csv`：逐样本“检索锚点位姿 / BEVPlace++ 3DoF 粗定位结果”对比结果
- `debug_outputs/batch_evaluate/range_0000_0019/summary.json`：整体统计摘要

如果希望批量导出每个 query 的可视化结果，可以加：

```bash
cd /home/qimao/grad_ws/bev_esti
python3 batch_evaluate.py \
  --start-index 10 \
  --limit 10 \
  --topk 5 \
  --device cpu \
  --visualize
```

此时会额外导出：

- `debug_outputs/batch_evaluate/range_0010_0019/visualizations/query_xxxx/topk_summary.png`
- `debug_outputs/batch_evaluate/range_0010_0019/visualizations/query_xxxx/rank_XX_db_xxxx_panel.png`
- `results.csv` 中也会记录每个 query 的可视化目录与总览图路径

---

## 8. 当前位姿恢复逻辑说明

当前 `bev_esti` 的位姿恢复逻辑参考自 `BEVPlace2` 后半段源码，主要流程为：

1. 对 query BEV 提取：
   - 全局描述子
   - 局部特征图
2. 用全局描述子在数据库中做 L2 搜索，得到 Top-K 候选
3. 对每个候选数据库图像：
   - 提取局部特征图
   - 用 FAST 提取关键点
   - 从局部特征图上采样关键点描述子
   - 用 BFMatcher 匹配描述子
   - 用 `rigid_ransac` 恢复 query 相对于数据库的 2D 刚体变换
4. 将相对变换与数据库锚点全局位姿相乘，恢复 query 的全局 `x, y, yaw`
5. 从 Top-K 中选出最优候选

默认优先使用：

- `inlier_count` 最大的候选
- 若 `inlier_count` 相同，则选全局描述子距离更近的候选

若所有候选都无法完成局部匹配，则退化为：

- 直接输出 Top-1 数据库锚点位姿作为粗定位结果

---

## 9. 需要注意的一个问题

当前项目中的 BEV 分辨率是：

- `0.2 m/pixel`

而原始 `BEVPlace2` 公开代码里的局部位姿恢复部分默认按：

- `0.4`

做像素到米的换算。

本仓库已经把位姿恢复的尺度换算改成：

- **优先读取数据库样本表中的 `bev_resolution_m`**

因此这里不会再写死 `0.4`。

---

## 10. 当前适用范围

这个仓库当前更适合：

- 单张 query 的 CPU 推理
- 本地调试和误差分析
- 检查训练模型是否真的能输出稳定的粗位姿
- 为后续 ICP 集成提供单样本接口

它当前不负责：

- 模型训练
- 大规模批量检索 benchmark
- ICP 精配准
- ROS2 在线节点

这些工作仍建议保留在主项目或后续单独模块中完成。

---

## 11. 一句话总结

`bev_esti` 的定位是：

> 一个面向本地 CPU 的轻量级 BEV 位姿估计仓库，用训练好的 `BEVPlace2` 模型对单张查询 BEV 图像完成“粗检索 + 3DoF 位姿恢复”，输出可直接用于后续 ICP 的初始位姿候选。
