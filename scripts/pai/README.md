# OpenPAI 提交 open-unlearning 实验 — 完整操作手册

## 一、平台信息

| 项目 | 地址/信息 |
|------|----------|
| **OpenPAI Web UI** | https://210.75.240.150/ |
| **Docker 仓库 (Harbor)** | https://210.75.240.150:30003/ |
| **开发机 (SSH)** | `liumingxuan@210.75.240.135` |
| **用户名** | `liumingxuan` |
| **Docker 镜像** | `210.75.240.150:30003/liumingxuan/open-unlearning:v1` |

## 二、核心概念：为什么需要共享存储

```
 你的开发机 (ubuntu135)               PAI 集群 (GPU 节点)
┌──────────────────────────┐        ┌──────────────────────────┐
│ ~/open-unlearning/       │        │  Docker 容器              │
│ ~/model/                 │  ✗     │   /workspace/open-...    │ ← 代码在镜像内
│  (PAI 节点看不到!)        │  不通   │                          │
│                          │        │                          │
│ /netdisk/liumingxuan/    │  ✓     │  /mnt/confignfs/         │
│   models/                │ ←NFS→  │   userdata/liumingxuan/  │ ← 模型、数据
│   data/                  │  共享   │   usercache/liumingxuan/ │ ← 缓存
│   saves/                 │        │                          │
└──────────────────────────┘        └──────────────────────────┘
```

**关键**: `~/model/` 和 `~/open-unlearning/data/` 只在开发机上。
PAI 节点只能访问 `/netdisk/` 和 `/netcache/`（通过 NFS 挂载到容器内的 `/mnt/confignfs/`）。

**路径映射**:
- 开发机 `/netdisk/liumingxuan/` = PAI 容器内 `/mnt/confignfs/userdata/liumingxuan/`
- 开发机 `/netcache/liumingxuan/` = PAI 容器内 `/mnt/confignfs/usercache/liumingxuan/`

## 三、GPU 资源

| SKU 名 | GPU | 显存 | CPU | 内存 | 适用场景 |
|--------|-----|------|-----|------|----------|
| `gpu-machine-3090` | 1× RTX 3090 | 24GB | 6 | 62.5GB | 7B 模型、单卡实验 |
| `gpu-machine-a100-lt` | 4× A100 | 4×80GB | 120 | 500GB | 大模型、多卡并行 |

| Virtual Cluster | 备注 |
|----------------|------|
| `default` | 默认集群 |
| `vc1` | 备用集群 |
| `vc2` | 备用集群 |

> 排队久的话可以切换 VC 试试。

---

## 四、完整流程（从零开始）

### Step 0: 首次环境准备（只做一次）

```bash
# 0.1 登录 Docker 仓库
docker login 210.75.240.150:30003 -u liumingxuan -p <密码>

# 0.2 创建共享存储目录
mkdir -p /netdisk/liumingxuan/{models,data,saves}
mkdir -p /netcache/liumingxuan/huggingface
```

### Step 1: 同步模型到共享存储

```bash
# 把本地模型复制到 netdisk（PAI 节点才能访问）
rsync -av --progress ~/model/Qwen_2.5-7B-Instruct/ \
    /netdisk/liumingxuan/models/Qwen2.5-7B-Instruct/

rsync -av --progress ~/model/llama2-7b-chat/llama2-7b-chat-hf/ \
    /netdisk/liumingxuan/models/Llama-2-7b-chat-hf/
```

> **每个新模型都要做这一步！** 否则 PAI 作业找不到模型文件。

### Step 2: 同步数据到共享存储

```bash
rsync -av ~/open-unlearning/data/edit/ /netdisk/liumingxuan/data/edit/
rsync -av ~/open-unlearning/data/tofu/ /netdisk/liumingxuan/data/tofu/
```

### Step 3: 构建 & 推送 Docker 镜像

```bash
cd ~/open-unlearning

# 构建（约 1-2 分钟，依赖层有缓存会更快）
docker build -t 210.75.240.150:30003/liumingxuan/open-unlearning:v2 \
    -f Dockerfile .

# 推送到仓库
docker push 210.75.240.150:30003/liumingxuan/open-unlearning:v2
```

> **改了代码就要重新构建！** 用新 tag（v2, v3...）。
> **只改了数据/模型不用重新构建。**

### Step 4: 编写作业 YAML

**方式 A — 用生成器（推荐）**:

```bash
# 知识编辑实验
python scripts/pai/gen_job.py --name rome-zsre --editor ROME -o job.yaml

# 指定模型
python scripts/pai/gen_job.py --name memit-test --editor MEMIT \
    --model Qwen2.5-7B-Instruct -o job.yaml

# 用 A100
python scripts/pai/gen_job.py --name grace-a100 --editor GRACE \
    --sku a100 -o job.yaml

# 自定义命令
python scripts/pai/gen_job.py --name my-exp \
    --cmd "python src/train.py --config-name=unlearn.yaml ..." -o job.yaml
```

**方式 B — 手写/改模板**:

复制 `scripts/pai/job_edit_rome.yaml`，修改 `name`、`commands`、`skuType` 等字段。

**方式 C — Web UI**:

打开 https://210.75.240.150/ → Submit Job → Single Job → 填表提交。

### Step 5: 提交作业

```bash
bash scripts/pai/submit.sh job.yaml
```

输出示例:
```
登录成功
作业名: rome-zsre
提交中...
提交成功! HTTP 202
查看: https://210.75.240.150/job-detail.html?username=liumingxuan&jobName=rome-zsre
```

### Step 6: 查看结果

1. **Web UI**: Jobs → 点击作业名 → **Stdout** 查看日志
2. **SSH 进容器**: 作业详情页有 SSH Info，可以 `ssh` 进去调试
3. **结果文件**: 在 `/netdisk/liumingxuan/saves/` 下（开发机上也能看到）

---

## 五、YAML 关键字段说明

```yaml
protocolVersion: 2
name: my-job-name              # 作业名（全局唯一）

prerequisites:
  - type: dockerimage
    uri: "210.75.240.150:30003/liumingxuan/open-unlearning:v1"  # 你的镜像
    name: docker_image0

taskRoles:
  taskrole:
    dockerImage: docker_image0
    resourcePerInstance:
      gpu: 1                   # GPU 数量
      cpu: 6                   # CPU 核心数
      memoryMB: 62500          # 内存 (MB)
    commands:                  # 要执行的命令（按顺序）
      - "cd /workspace/open-unlearning"
      - "python src/train.py ..."

defaults:
  virtualCluster: default      # 可选: default, vc1, vc2

extras:
  com.microsoft.pai.runtimeplugin:
    - plugin: teamwise_storage # 挂载共享存储（必需！）
      parameters:
        storageConfigNames:
          - usercache           # → /mnt/confignfs/usercache/
          - userdata            # → /mnt/confignfs/userdata/
```

---

## 六、日常操作速查

### 只改了代码

```bash
docker build -t 210.75.240.150:30003/liumingxuan/open-unlearning:v2 -f Dockerfile .
docker push 210.75.240.150:30003/liumingxuan/open-unlearning:v2
# 作业 YAML 里 uri 改成 :v2
```

### 添加新模型

```bash
rsync -av ~/model/新模型/ /netdisk/liumingxuan/models/新模型/
```

### 查看排队原因

```bash
# 查看集群资源使用
bash scripts/pai/submit.sh  # 不带参数会报错但可以看登录是否正常

# 或直接去 Web UI → Jobs 页面查看
```

### 停止/删除作业

在 Web UI 作业详情页点 **Stop** 按钮。

---

## 七、文件清单

```
open-unlearning/
├── Dockerfile                    # Docker 镜像定义
├── .dockerignore                 # 排除大文件（saves/、data/ 等）
└── scripts/pai/
    ├── README.md                 # 本文档
    ├── submit.sh                 # 命令行一键提交
    ├── gen_job.py                # YAML 生成器
    ├── build_and_push.sh         # 构建并推送镜像
    ├── prepare_storage.sh        # 同步模型/数据到共享存储
    ├── job_test_env.yaml         # 示例：环境检测
    ├── job_edit_rome.yaml        # 示例：ROME 知识编辑
    └── job_tofu_llama2_all.yaml  # 示例：TOFU 全算法 unlearning
```

## 八、常见问题

| 问题 | 原因 & 解决 |
|------|-------------|
| 作业一直 Waiting | GPU 不足，排队中。试试换 VC（`vc1`/`vc2`）或等待 |
| 找不到模型文件 | 没同步到 `/netdisk/`。执行 `rsync` 同步 |
| `import` 报错找不到模块 | 镜像代码过时。重新 `docker build` + `push` |
| 409 Conflict | 作业名已存在。改个名字重新提交 |
| 显存不足 OOM | 减小 `per_device_train_batch_size`，或换 A100 |
| 作业名乱码 | YAML 中 `name` 字段只能用英文、数字、`-` |
