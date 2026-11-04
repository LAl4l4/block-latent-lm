# 使用与复现

所有命令在项目根目录执行，当前可用环境为 `D:\Code\venv313\Scripts\python.exe`（Python 3.13.9，PyTorch 2.11.0+cu128）。数据已准备好，不需重新下载。

## 从已恢复的第 3100 步继续

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/main/last.pt
```

从 step 3101 继续，恢复模型、优化器和随机状态，总计划仍为 10000 步。自动保存每 100 步一次。第一次 Ctrl+C 请求完成当前更新后保存退出；强制结束进程不能保证完成当前保存。

保存遇到 Windows 临时读锁会重试；持续锁定则另存为 `last-step<step>.pt`，终端会显示真实文件名。此时下次用该版本文件 `--resume`，不要使用落后的 `last.pt`。

本次第 3100 步来自完整的 `last.tmp`，模型、优化器和 RNG 已验证。不可变备份为 `artifacts/checkpoints/main-step3100.pt`。恢复记录为 `runs/main/recovery.json`。整理过程中没有启动主模型训练。

## 验证与生成

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py `
  --resume runs/main/last.pt --eval-only --eval-batches 4

& D:\Code\venv313\Scripts\python.exe generate.py `
  --checkpoint runs/main/last.pt `
  --prompt "Once upon a time, a little girl found a strange box." `
  --max-new-tokens 64 --seed 42 --show-info
```

主干学习诊断：

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.diagnose_backbone_learning `
  --checkpoint runs/prior-block-head-verified/last.pt `
  --initial runs/prior-energy-v4-check/last.pt `
  --output runs/backbone-recheck.json
```

## 新建实验

默认 `configs/prior.json` 已是当前小输出头。以下命令新建实验，不继承第 3100 步的训练状态：

```powershell
& D:\Code\venv313\Scripts\python.exe train_prior.py --run-dir runs/new-experiment
```

只有重建 codec 时才需要 `train_codec.py`。原 codec 保留在 `runs/codec-v3/last.pt`。旧对照配置在 `paper/archive/configs/`，离线脚本统一以 `python -m experiments.<name>` 运行。

## 测试与环境

```powershell
$env:PYTHONPATH = Join-Path $PWD '.tmp/dev-tools'
$env:PYTHONDONTWRITEBYTECODE = '1'
& D:\Code\venv313\Scripts\python.exe -m pytest tests -q -p no:cacheprovider `
  --basetemp .tmp/pytest-paper-check
```

`.tmp/dev-tools` 暂保留本机的 pytest 等测试依赖，未加入 Git；旧 Python 3.12 `.venv` 已移除。依赖版本以 `paper/environment.json` 为训练事实记录，`requirements.txt` 保留运行和数据准备依赖；当前训练解释器未安装 pyarrow，但已准备数据的训练不依赖它。原 3.12 环境锁文件仅在 `paper/archive/` 作历史记录。

## Git 快照和归档恢复

`1763f0f` 是整理前的完整研究快照，数据和权重通过 Git LFS 保存。本项目没有推送到远端；本地 `.git/lfs` 是归档的一部分。活动 `runs/` 和 `data/` 不再跟踪，避免 Git 状态检查读取训练中的检查点。

归档文件的哈希、大小和历史位置记录在 `paper/artifacts.json`。需要旧实验时恢复到新路径，不重新跟踪活动训练文件：

```powershell
& D:\Code\venv313\Scripts\python.exe -m experiments.restore_artifact `
  runs/prior-mixture-v5-check/last.pt --output runs/restored-mixture/last.pt
```

恢复器验证本地快照对象的 SHA-256，并拒绝覆盖已有输出。历史报告中的路径和数值保留原样；当前脚本位置以本说明为准。
