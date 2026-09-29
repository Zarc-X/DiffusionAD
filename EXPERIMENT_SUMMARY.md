# 近期实验总结

## 1. 统计口径与数据来源

- 15 类全量口径：carpet, grid, leather, tile, wood, bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper。
- 3 类口径：carpet, screw, transistor。
- 汇总文件：
  - `outputs/metrics/ARGS=1/200_400t_1_MVTec_image_pixel_auroc_train.csv`
  - `outputs_mamba_recon/metrics/ARGS=mamba_low_unidir/200_400t_1_MVTec_image_pixel_auroc_train_mamba.csv`
  - `outputs_mamba_recon/metrics/ARGS=mamba_low_unidir_singlepath/200_400t_1_MVTec_image_pixel_auroc_train_mamba.csv`
  - `outputs_mamba_recon/metrics/ARGS=mamba_low_unidir_singlepath_resdistill/200_400t_1_MVTec_image_pixel_auroc_train_mamba.csv`
  - `outputs_mamba_recon/metrics/ARGS=mamba_low_unidir_singlepath_consistency/200_400t_1_MVTec_image_pixel_auroc_train_mamba.csv`

## 2. 阶段时间线与模型改动

### 阶段 A：全量测试起点（发现速度慢于原论文）

**模型改动**
- 使用 mamba low + unidir 重建骨干（双路径扩散默认逻辑仍在）。

**结果（15 类均值）**

| 配置 | Image-AUROC | Pixel-AUROC | Image-AP | Pixel-AP | Image-F1 | Pixel-F1 | Eval-FPS | ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原论文基线（paper_orig） | 98.893 | 98.060 | 99.573 | 74.907 | 98.340 | 70.727 | 10.691 | 97.588 |
| mamba_low_unidir | 96.153 | 97.740 | 98.320 | 69.733 | 96.107 | 66.513 | 10.348 | 97.153 |

**结果（15 类按类别展开，paper_orig -> mamba_low_unidir）**

| 类别 | A Image-AUROC | B Image-AUROC | Δ | A Pixel-AUROC | B Pixel-AUROC | Δ | A Pixel-AP | B Pixel-AP | Δ | A Pixel-F1 | B Pixel-F1 | Δ | A FPS | B FPS | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carpet | 99.200 | 90.400 | -8.800 | 99.300 | 97.500 | -1.800 | 82.800 | 70.000 | -12.800 | 75.200 | 65.500 | -9.700 | 7.876 | 11.238 | +3.361 |
| grid | 100.000 | 100.000 | +0.000 | 99.600 | 99.400 | -0.200 | 70.600 | 64.100 | -6.500 | 64.600 | 59.400 | -5.200 | 12.548 | 9.349 | -3.199 |
| leather | 100.000 | 100.000 | +0.000 | 99.700 | 99.600 | -0.100 | 68.500 | 58.600 | -9.900 | 67.800 | 61.600 | -6.200 | 11.989 | 9.971 | -2.018 |
| tile | 100.000 | 99.900 | -0.100 | 99.600 | 99.600 | +0.000 | 96.200 | 95.800 | -0.400 | 89.200 | 89.000 | -0.200 | 12.656 | 11.168 | -1.488 |
| wood | 99.800 | 99.300 | -0.500 | 98.300 | 96.200 | -2.100 | 86.900 | 78.200 | -8.700 | 79.600 | 72.600 | -7.000 | 11.774 | 10.291 | -1.484 |
| bottle | 99.400 | 93.400 | -6.000 | 98.800 | 97.900 | -0.900 | 88.400 | 78.200 | -10.200 | 80.700 | 71.500 | -9.200 | 12.497 | 10.863 | -1.634 |
| cable | 97.100 | 96.900 | -0.200 | 96.500 | 97.500 | +1.000 | 61.100 | 64.900 | +3.800 | 60.000 | 64.200 | +4.200 | 7.695 | 10.191 | +2.496 |
| capsule | 93.900 | 84.500 | -9.400 | 98.200 | 97.000 | -1.200 | 52.500 | 42.600 | -9.900 | 53.300 | 45.200 | -8.100 | 7.275 | 9.140 | +1.865 |
| hazelnut | 100.000 | 100.000 | +0.000 | 99.700 | 99.300 | -0.400 | 91.300 | 86.100 | -5.200 | 84.200 | 78.300 | -5.900 | 11.909 | 8.798 | -3.111 |
| metal_nut | 99.900 | 100.000 | +0.100 | 99.500 | 98.700 | -0.800 | 94.800 | 90.300 | -4.500 | 91.600 | 85.600 | -6.000 | 10.912 | 10.484 | -0.428 |
| pill | 97.400 | 97.800 | +0.400 | 99.300 | 99.300 | +0.000 | 86.700 | 87.000 | +0.300 | 79.200 | 79.300 | +0.100 | 12.812 | 10.842 | -1.970 |
| screw | 98.000 | 90.700 | -7.300 | 98.700 | 98.100 | -0.600 | 62.300 | 55.800 | -6.500 | 60.300 | 55.400 | -4.900 | 8.284 | 10.417 | +2.133 |
| toothbrush | 100.000 | 98.600 | -1.400 | 98.500 | 96.300 | -2.200 | 60.100 | 48.200 | -11.900 | 58.900 | 49.600 | -9.300 | 12.228 | 11.313 | -0.915 |
| transistor | 98.800 | 90.800 | -8.000 | 86.600 | 90.800 | +4.200 | 49.800 | 56.100 | +6.300 | 50.000 | 55.200 | +5.200 | 8.361 | 10.490 | +2.129 |
| zipper | 99.900 | 100.000 | +0.100 | 98.600 | 98.900 | +0.300 | 71.600 | 70.100 | -1.500 | 66.300 | 65.300 | -1.000 | 11.546 | 10.665 | -0.881 |

**对比结论**
- 速度：FPS 下降 0.343（约 -3.2%），对应“全量测试比原论文慢”的问题起点。
- 精度：Pixel-AP 下降 5.174，Pixel-F1 下降 4.214。

---

### 阶段 B：Single-path v1（self proxy）

**模型改动**
- 引入单路径近似：`diffusion_single_path=1`。
- 代理模式 `single_path_proxy_mode=self`，不再显式计算第二条高噪分支。
- 训练损失通过 `single_path_loss_scale` 做近似补偿。

**结果（15 类 first 口径）**

| 配置 | Image-AUROC | Pixel-AUROC | Image-AP | Pixel-AP | Image-F1 | Pixel-F1 | Eval-FPS | ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mamba_low_unidir | 96.153 | 97.740 | 98.320 | 69.733 | 96.107 | 66.513 | 10.348 | 97.153 |
| singlepath v1（first） | 97.960 | 97.693 | 99.227 | 72.193 | 97.420 | 68.267 | 18.832 | 53.299 |

**结果（15 类按类别展开，mamba_low_unidir -> singlepath v1 first）**

| 类别 | A Image-AUROC | B Image-AUROC | Δ | A Pixel-AUROC | B Pixel-AUROC | Δ | A Pixel-AP | B Pixel-AP | Δ | A Pixel-F1 | B Pixel-F1 | Δ | A FPS | B FPS | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carpet | 90.400 | 99.000 | +8.600 | 97.500 | 98.700 | +1.200 | 70.000 | 78.400 | +8.400 | 65.500 | 72.400 | +6.900 | 11.238 | 18.651 | +7.413 |
| grid | 100.000 | 100.000 | +0.000 | 99.400 | 99.600 | +0.200 | 64.100 | 70.000 | +5.900 | 59.400 | 65.400 | +6.000 | 9.349 | 18.113 | +8.764 |
| leather | 100.000 | 100.000 | +0.000 | 99.600 | 99.500 | -0.100 | 58.600 | 56.700 | -1.900 | 61.600 | 58.400 | -3.200 | 9.971 | 18.867 | +8.895 |
| tile | 99.900 | 100.000 | +0.100 | 99.600 | 99.500 | -0.100 | 95.800 | 95.500 | -0.300 | 89.000 | 88.500 | -0.500 | 11.168 | 20.109 | +8.941 |
| wood | 99.300 | 98.600 | -0.700 | 96.200 | 96.200 | +0.000 | 78.200 | 78.700 | +0.500 | 72.600 | 72.900 | +0.300 | 10.291 | 20.134 | +9.844 |
| bottle | 93.400 | 98.800 | +5.400 | 97.900 | 98.800 | +0.900 | 78.200 | 84.900 | +6.700 | 71.500 | 78.200 | +6.700 | 10.863 | 18.136 | +7.273 |
| cable | 96.900 | 96.200 | -0.700 | 97.500 | 97.500 | +0.000 | 64.900 | 70.500 | +5.600 | 64.200 | 65.100 | +0.900 | 10.191 | 18.833 | +8.642 |
| capsule | 84.500 | 91.300 | +6.800 | 97.000 | 97.500 | +0.500 | 42.600 | 48.300 | +5.700 | 45.200 | 50.200 | +5.000 | 9.140 | 18.917 | +9.777 |
| hazelnut | 100.000 | 100.000 | +0.000 | 99.300 | 99.300 | +0.000 | 86.100 | 84.400 | -1.700 | 78.300 | 76.700 | -1.600 | 8.798 | 19.942 | +11.145 |
| metal_nut | 100.000 | 100.000 | +0.000 | 98.700 | 99.500 | +0.800 | 90.300 | 95.900 | +5.600 | 85.600 | 92.000 | +6.400 | 10.484 | 17.350 | +6.866 |
| pill | 97.800 | 98.100 | +0.300 | 99.300 | 99.000 | -0.300 | 87.000 | 82.900 | -4.100 | 79.300 | 73.900 | -5.400 | 10.842 | 20.068 | +9.226 |
| screw | 90.700 | 89.900 | -0.800 | 98.100 | 98.100 | +0.000 | 55.800 | 58.400 | +2.600 | 55.400 | 57.300 | +1.900 | 10.417 | 20.513 | +10.096 |
| toothbrush | 98.600 | 100.000 | +1.400 | 96.300 | 97.600 | +1.300 | 48.200 | 56.400 | +8.200 | 49.600 | 55.500 | +5.900 | 11.313 | 16.414 | +5.101 |
| transistor | 90.800 | 97.800 | +7.000 | 90.800 | 85.600 | -5.200 | 56.100 | 44.900 | -11.200 | 55.200 | 46.200 | -9.000 | 10.490 | 18.630 | +8.140 |
| zipper | 100.000 | 99.700 | -0.300 | 98.900 | 99.000 | +0.100 | 70.100 | 77.000 | +6.900 | 65.300 | 71.300 | +6.000 | 10.665 | 17.796 | +7.131 |

**对比结论**
- 速度显著提升：FPS +8.484，时延下降 43.854 ms。
- 精度整体回升：Pixel-AP +2.460，Pixel-F1 +1.754（相对 low_unidir）。

---

### 阶段 C：Single-path v2（pseudo_noisier proxy）

**模型改动**
- 单路径代理升级为 pseudo_noisier：在高时间步生成伪更噪样本再回投。
- 关键参数为 `single_path_delta_t`、`single_path_proxy_mix`。

**结果（3 类同口径：singlepath first -> latest）**

| 3 类均值对比 | Image-AUROC | Pixel-AUROC | Image-AP | Pixel-AP | Image-F1 | Pixel-F1 | Eval-FPS | ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| singlepath 3 类 first | 95.567 | 94.133 | 97.900 | 60.567 | 93.267 | 58.633 | 19.264 | 52.015 |
| singlepath 3 类 latest | 92.167 | 92.400 | 96.867 | 51.967 | 91.267 | 50.900 | 10.723 | 93.421 |
| 差值（latest-first） | -3.400 | -1.733 | -1.033 | -8.600 | -2.000 | -7.733 | -8.541 | +41.406 |

**结果（3 类按类别展开，singlepath first -> singlepath latest）**

| 类别 | A Image-AUROC | B Image-AUROC | Δ | A Pixel-AUROC | B Pixel-AUROC | Δ | A Pixel-AP | B Pixel-AP | Δ | A Pixel-F1 | B Pixel-F1 | Δ | A FPS | B FPS | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carpet | 99.000 | 92.200 | -6.800 | 98.700 | 97.600 | -1.100 | 78.400 | 65.200 | -13.200 | 72.400 | 60.700 | -11.700 | 18.651 | 10.111 | -8.539 |
| screw | 89.900 | 86.500 | -3.400 | 98.100 | 97.900 | -0.200 | 58.400 | 53.300 | -5.100 | 57.300 | 53.900 | -3.400 | 20.513 | 10.937 | -9.576 |
| transistor | 97.800 | 97.800 | +0.000 | 85.600 | 81.700 | -3.900 | 44.900 | 37.400 | -7.500 | 46.200 | 38.100 | -8.100 | 18.630 | 11.120 | -7.510 |

**对比结论**
- v2 让 single-path 的速度优势基本消失，同时精度也显著回退。
- 尤其像素级指标（Pixel-AP / Pixel-F1）下降明显。

---

### 阶段 D：Residual Distill

**模型改动**
- 引入 residual distill：训练时用 teacher residual 稀疏监督 student residual。
- 推理仍走单路径，不引入额外推理分支。

**结果（3 类，vs singlepath latest）**

| 3 类均值对比 | Image-AUROC | Pixel-AUROC | Image-AP | Pixel-AP | Image-F1 | Pixel-F1 | Eval-FPS | ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| singlepath latest | 92.167 | 92.400 | 96.867 | 51.967 | 91.267 | 50.900 | 10.723 | 93.421 |
| resdistill | 91.800 | 90.700 | 96.000 | 51.700 | 89.900 | 50.967 | 19.598 | 51.180 |
| 差值（resdistill-singlepath latest） | -0.367 | -1.700 | -0.867 | -0.267 | -1.367 | +0.067 | +8.875 | -42.241 |

**结果（3 类按类别展开，singlepath latest -> resdistill）**

| 类别 | A Image-AUROC | B Image-AUROC | Δ | A Pixel-AUROC | B Pixel-AUROC | Δ | A Pixel-AP | B Pixel-AP | Δ | A Pixel-F1 | B Pixel-F1 | Δ | A FPS | B FPS | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carpet | 92.200 | 86.200 | -6.000 | 97.600 | 97.700 | +0.100 | 65.200 | 68.700 | +3.500 | 60.700 | 64.200 | +3.500 | 10.111 | 18.138 | +8.027 |
| screw | 86.500 | 94.900 | +8.400 | 97.900 | 98.400 | +0.500 | 53.300 | 53.600 | +0.300 | 53.900 | 52.200 | -1.700 | 10.937 | 20.056 | +9.120 |
| transistor | 97.800 | 94.300 | -3.500 | 81.700 | 76.000 | -5.700 | 37.400 | 32.800 | -4.600 | 38.100 | 36.500 | -1.600 | 11.120 | 20.599 | +9.479 |

**对比结论**
- 主要价值在速度恢复：FPS 大幅回升。
- 精度总体未形成稳定收益，接近持平或小幅回退。

---

### 阶段 E：Consistency（双代理一致性）

**模型改动**
- 在主 residual 上叠加样本级一致性权重：
  - 主 residual 与第二代理 residual 做相似性；
  - 相似性映射为权重后缩放引导项。

**结果（3 类，vs resdistill）**

| 3 类均值对比 | Image-AUROC | Pixel-AUROC | Image-AP | Pixel-AP | Image-F1 | Pixel-F1 | Eval-FPS | ms/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resdistill | 91.800 | 90.700 | 96.000 | 51.700 | 89.900 | 50.967 | 19.598 | 51.180 |
| consistency | 90.167 | 91.600 | 94.733 | 52.167 | 88.133 | 50.833 | 10.407 | 96.503 |
| 差值（consistency-resdistill） | -1.633 | +0.900 | -1.267 | +0.467 | -1.767 | -0.134 | -9.191 | +45.323 |

**结果（3 类按类别展开，resdistill -> consistency）**

| 类别 | A Image-AUROC | B Image-AUROC | Δ | A Pixel-AUROC | B Pixel-AUROC | Δ | A Pixel-AP | B Pixel-AP | Δ | A Pixel-F1 | B Pixel-F1 | Δ | A FPS | B FPS | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carpet | 86.200 | 88.000 | +1.800 | 97.700 | 97.500 | -0.200 | 68.700 | 66.800 | -1.900 | 64.200 | 61.800 | -2.400 | 18.138 | 11.311 | -6.827 |
| screw | 94.900 | 91.200 | -3.700 | 98.400 | 98.200 | -0.200 | 53.600 | 53.300 | -0.300 | 52.200 | 52.800 | +0.600 | 20.056 | 9.642 | -10.414 |
| transistor | 94.300 | 91.300 | -3.000 | 76.000 | 79.100 | +3.100 | 32.800 | 36.400 | +3.600 | 36.500 | 37.900 | +1.400 | 20.599 | 10.268 | -10.331 |

**对比结论**
- 像素侧部分指标略有改善（Pixel-AUROC、Pixel-AP）。
- 但图像级指标和速度明显回退，综合性价比不如 resdistill。

---

## 3. 总结

- 当前链路里，单看“速度+精度综合折中”：
  - Single-path v1 是最明显的正向拐点；
  - v2（pseudo_noisier）导致速度与精度双回退；
  - resdistill 把速度拉回来了，但精度收益有限；
  - consistency 目前版本没有形成更优综合解。
- 关键经验：
  - 额外代理/额外前向很容易吞掉 single-path 的速度红利；
  - transistor 与 screw 仍是决定三类均值波动的核心难类。

## 4. 正在进行中的实验

### 时间步分段引导调度（timestep schedule）

**模型改动（已接入）**
- 新增分段调度开关与参数：
  - `guidance_timestep_schedule_enable`
  - `guidance_timestep_schedule_t1` / `guidance_timestep_schedule_t2`
  - `guidance_timestep_schedule_w_early` / `guidance_timestep_schedule_w_mid` / `guidance_timestep_schedule_w_late`
- 已在训练和评估的一步引导路径接入。

**本轮实验配置**
- `args/args_mamba_low_unidir_singlepath_tschedule.json`
- 当前设置：开启 schedule，关闭 consistency，关闭 IRF，关闭 distill，用于做干净 A/B。

**记录状态**
- 本章仅记录“正在实验”，结果待实验跑完后补充。

## 5. 模型改动机制详解（补充）

### 5.1 Single-path v1（self proxy）引导项

Single-path v1 的核心是：不再显式算第二条 noisier 分支，而是在 normal 分支内部自构造 proxy。

1. 先得到 normal 分支噪声预测
- 在 normal_t 上做一次扩散损失，得到 `x_normal_t` 和 `estimate_noise_normal`。

2. 反推 `x0` 再回投到同一个 `t`
- `pred_x_0_normal = predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_normal)`
- `pred_x_t_self = sample_q(pred_x_0_normal, normal_t, estimate_noise_normal)`

3. 用 proxy 与当前噪声态做残差，引导噪声预测
- `residual = (pred_x_t_self - x_normal_t) * single_path_residual_scale`
- `estimate_noise_hat = estimate_noise_normal - sqrt(1-alpha_bar_t) * guidance_scale * residual`

其中 `guidance_scale` 在不启用分段调度时可理解为：

`guidance_scale = condition_w * single_path_condition_w_scale`

结论：v1 几乎不增加额外前向，速度友好。

### 5.2 Single-path v2（pseudo_noisier）

v2 在 v1 基础上增加了一条“伪高噪时间步”路径：

1. 构造更高时间步
- `high_t = clamp(normal_t + delta_t)`

2. 从 `pred_x_0_normal` 合成伪更噪样本
- `x_pseudo_noisier_t = sample_q(pred_x_0_normal, high_t, estimate_noise_normal)`

3. 在 `high_t` 上再做一次模型前向
- `estimate_noise_high = model(x_pseudo_noisier_t, high_t)`
- `pred_x_0_high = predict_x_0_from_eps(x_pseudo_noisier_t, high_t, estimate_noise_high)`

4. 把 high_t 信息回投到 normal_t
- `pred_x_t_high = sample_q(pred_x_0_high, normal_t, estimate_noise_normal)`

5. 与 self proxy 混合形成最终 proxy
- `pred_x_t_proxy = (1 - proxy_mix) * pred_x_t_self + proxy_mix * pred_x_t_high`

结论：v2 的主要代价来自第 3 步额外前向，因此很容易吞掉 v1 的速度优势。

### 5.3 Residual Distill（稀疏 teacher residual 监督）

Residual Distill 只发生在训练期，推理路径保持 single-path。

1. 稀疏触发条件
- `single_path_residual_distill` 必须开启。
- `single_path_distill_weight > 0`。
- 按 `single_path_distill_prob` 进行概率触发（不是每个 iteration 都算）。

2. 构造 teacher residual（`no_grad`）
- 采样 `noisier_t`（由 `single_path_distill_use_delta_t` / `single_path_distill_min_t` 决定）。
- 在 `noisier_t` 上得到 teacher 路径：
  - `x_noisier_t -> estimate_noise_noisier -> pred_x_0_noisier -> pred_x_t_teacher`
- `teacher_residual = pred_x_t_teacher - x_normal_t`

3. 构造 student residual
- `student_residual = pred_x_t_proxy - x_normal_t`

4. 归一化与蒸馏损失
- 可选按 `sqrt(1-alpha_bar_t)` 归一化（`single_path_distill_normalize`）。
- `distill_loss = mean((student_residual - teacher_residual)^2)`
- 只在 normal 样本上聚合（`anomaly_label == 0`）。

5. 合并到训练总损失
- `loss = normal_noise_loss + single_path_distill_weight * distill_loss`

结论：蒸馏分支让训练更“像双路径”，但推理仍是单路径，因此目标是尽量保速度、补精度。

### 5.4 Consistency（双代理一致性样本级加权）

Consistency 的核心是：先算主 residual，再用“主 residual 与第二代理 residual 的一致性”生成样本级权重，最后缩放主 residual。

1. 主 residual
- `residual_1 = (pred_x_t_proxy - x_normal_t) * single_path_residual_scale`

2. 第二代理 residual
- 用 `single_path_consistency_proxy_mode`（常用 pseudo_noisier）构造第二代理 `pred_x_t_proxy_2`。
- `residual_2 = (pred_x_t_proxy_2 - x_normal_t) * single_path_consistency_residual_scale`

3. 一致性分数（样本级）
- 将 residual 展平后做 cosine：`cos = cosine_similarity(residual_1, residual_2)`
- 映射到 `[0,1]`：`gate0 = (cos + 1) / 2`
- 再做温度门控：`gate = sigmoid((gate0 - tau) * beta)`

4. 生成样本级权重并缩放主 residual
- `weight = min_w + (max_w - min_w) * gate`
- `residual = residual_1 * weight`

注意：这里的 weight 是每张图一个标量（样本级），不是像素级权重图。

### 5.5 时间步分段引导调度（timestep schedule）

分段调度不是新分支，它只改“引导强度随时间步 t 的权重”。

1. 三段权重定义
- early：`t <= t1`，权重 `w_early`
- middle：`t1 < t < t2`，权重 `w_mid`
- late：`t >= t2`，权重 `w_late`

2. 组装分段权重
- 为 batch 内每个样本按其 `normal_t` 选取段权重，形成样本级 `segment_weight`。

3. 与基础引导强度相乘
- `guidance_scale = base_scale * segment_weight`
- single-path 下常见 `base_scale = condition_w * single_path_condition_w_scale`

4. 最终作用位置
- 在 `estimate_noise_hat` 的引导项里生效：
  - `estimate_noise_hat = estimate_noise_normal - sqrt(1-alpha_bar_t) * guidance_scale * residual`

5. 与其他改动的关系
- schedule 可以独立启用，也可与 single-path/resdistill/consistency叠加。
- 它不引入额外模型前向，理论上开销很小，主要影响“不同噪声阶段引导强弱分配”。

当前正在实验的配置就是仅开启 schedule、关闭 consistency/distill，用于干净评估 schedule 本身收益。
