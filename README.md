# 块级并行语言模型：衔接处的依赖该怎么丢

本仓库研究块级（block）并行文本生成：几个块同时生成时，必须丢掉一部分词之间的依赖，丢哪些最划算。研究分三个阶段，每个阶段的代码、脚本、测试和文档都放在各自的文件夹里。

| 阶段 | 时间 | 内容 | 状态 |
| --- | --- | --- | --- |
| [stage1](stage1/README.md) | 10-28 至 10-31 | 块级潜变量系统：主干 B、局部解码器 D、历史编码器 E、反馈学生 P | 已停止，只留文档；代码在 tag `stage1-latent-blocks` |
| [stage2](stage2/README.md) | 10-31 至 11-04 | 倒序错开解码：把每块的首词放到最后写 | 论文已完成，待投 arXiv |
| [stage3](stage3/README.md) | 11-01 起 | 三级解码：大 latent → 联合采样的块 latent → 倒序错开解码 | 进行中 |

## 目录

| 路径 | 内容 |
| --- | --- |
| `common/` | 各阶段共用的代码：Transformer 组件、训练工具、定长窗口数据 |
| `data_prep/` | TinyStories 下载与预处理，以及数据卡 |
| `stage1/` | 阶段 1 的文档 |
| `stage2/` | 阶段 2 的代码（`mask/`、`plan/`）、脚本、测试、文档和论文（`paper/`） |
| `stage3/` | 阶段 3 的代码（`latent/`、`window/`）、脚本、测试和文档 |
| `timestamps/` | 想法提出时间的 OpenTimestamps 证明，不要改动 |
| `runs/`、`data/` | 训练结果与数据，只存在本地，不进 Git |

## 运行

所有命令都在仓库根目录执行，各阶段的具体命令见各自的 README。

```powershell
# 数据（输出到 data/raw 和 data/prepared）
& D:\Code\venv313\Scripts\python.exe -m data_prep.download
& D:\Code\venv313\Scripts\python.exe -m data_prep.prepare

# 例子：阶段 2 的 mask 实验
& D:\Code\venv313\Scripts\python.exe -m stage2.mask.train --schedule reverse --stagger 8 --segment 8

# 全部测试
& D:\Code\venv313\Scripts\python.exe -m pytest
```

## 仓库说明

- 远程仓库：`github.com/LAl4l4/block-latent-lm`，目前为私有。
- 阶段 1 的代码（`latent_ar/`、`experiments/`、根目录的 `train*.py`、`configs/`、`artifacts/`）在 2026-11-04 整理时删除，完整保留在 tag `stage1-latent-blocks`：`git checkout stage1-latent-blocks` 即可取回。
- 2026-10-31 推送前，所有大文件（`*.pt`、`*.bin`、`*.npy`、`*.parquet`、准备好的数据）已从 Git 历史中删除，只保留在本地。提交的时间和内容不变，但哈希值已改变：旧文档中提到的 `1763f0f` 现在对应 `45a20d2`。
