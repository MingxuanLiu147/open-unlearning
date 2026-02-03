# Git 完整提交流程指南

## 一、首次设置（只需执行一次）

### 1. 检查 SSH 密钥是否存在
```bash
ls -la ~/.ssh/id_*.pub
```

### 2. 如果没有 SSH 密钥，生成新的密钥对
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# 或者使用 RSA
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# 按 Enter 使用默认路径，可以设置密码或留空
```

### 3. 查看并复制 SSH 公钥
```bash
cat ~/.ssh/id_rsa.pub
# 或
cat ~/.ssh/id_ed25519.pub
```

### 4. 将 SSH 公钥添加到 GitHub
1. 访问：https://github.com/settings/keys
2. 点击 "New SSH key"
3. 填写 Title（如：My Linux Server）
4. 粘贴公钥内容
5. 点击 "Add SSH key"

### 5. 测试 SSH 连接
```bash
ssh -T git@github.com
# 应该看到：Hi [username]! You've successfully authenticated...
```

### 6. Fork 原始仓库（如果还没有）
1. 访问原始仓库：https://github.com/locuslab/open-unlearning
2. 点击右上角 "Fork" 按钮
3. Fork 到你的账户：`https://github.com/你的用户名/open-unlearning`

### 7. 配置远程仓库地址（使用 SSH）
```bash
cd ~/open-unlearning
git remote -v  # 查看当前远程地址

# 如果指向原始仓库，改为你的 fork（使用 SSH）
git remote set-url origin git@github.com:你的用户名/open-unlearning.git

# 验证修改
git remote -v
```

---

## 二、日常提交流程

### 步骤 1：查看变更
```bash
cd ~/open-unlearning
git status                    # 查看文件状态
git diff                      # 查看详细变更（如果输出很长，按 'q' 退出）
```

### 步骤 2：暂存变更
```bash
# 暂存所有变更
git add .

# 或者只暂存特定文件
git add 文件路径1 文件路径2

# 查看暂存的文件
git status
```

### 步骤 3：提交变更
```bash
git commit -m "提交信息标题

详细说明：
- 变更1
- 变更2
- 变更3"
```

**提交信息示例：**
```bash
git commit -m "优化显存配置并修复代码健壮性问题

### 步骤 4：拉取远程最新更改（避免冲突）
```bash
# 先获取远程更新
git fetch origin

# 查看远程和本地的差异
git log HEAD..origin/main --oneline  # 远程有但本地没有的提交
git log origin/main..HEAD --oneline  # 本地有但远程没有的提交

# 使用 rebase 方式合并（推荐，保持历史整洁）
git pull --rebase origin main

# 或者使用 merge 方式
git pull origin main
```

**如果 rebase 过程中有冲突：**
```bash
# 1. 解决冲突后
git add 冲突文件
git rebase --continue

# 2. 如果想取消 rebase
git rebase --abort
```

### 步骤 5：推送到远程仓库
```bash
git push
```

**如果推送被拒绝（远程有新的提交）：**
```bash
# 先拉取并合并
git pull --rebase origin main
# 然后再推送
git push
```

**如果本地分支名不是 main：**
```bash
git push -u origin 分支名
```

### 步骤 6：验证推送成功
```bash
git status
# 应该看到：Your branch is up to date with 'origin/main'
```

---

## 三、常见问题处理

### 问题 1：权限被拒绝（403 错误）

**原因：** 使用 HTTPS 但没有配置认证

**解决方案：**
```bash
# 方法1：改用 SSH（推荐）
git remote set-url origin git@github.com:你的用户名/仓库名.git

# 方法2：使用 Personal Access Token
# 1. 访问：https://github.com/settings/tokens
# 2. 生成新的 token（选择 repo 权限）
# 3. 推送时使用 token 作为密码
git push
# Username: 你的用户名
# Password: 你的 token（不是密码）
```

### 问题 2：推送被拒绝（远程有新的提交）

**解决方案：**
```bash
# 先拉取远程更改
git pull --rebase origin main
# 解决可能的冲突后
git push
```

### 问题 3：想撤销最后一次提交

```bash
# 只撤销提交，保留文件修改
git reset --soft HEAD~1

# 撤销提交并取消暂存，但保留文件修改
git reset HEAD~1

# 完全撤销提交和修改（危险！）
git reset --hard HEAD~1
```

### 问题 4：修改最后一次提交信息

```bash
git commit --amend -m "新的提交信息"
```

### 问题 5：查看提交历史

```bash
git log --oneline          # 简洁显示
git log --graph --oneline  # 图形化显示
git log -5                 # 显示最近5次提交
```

---

## 四、完整流程示例

```bash
# ========== 首次设置（只需一次） ==========
# 1. 生成 SSH 密钥（如果没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 复制公钥并添加到 GitHub
cat ~/.ssh/id_ed25519.pub

# 3. 测试 SSH 连接
ssh -T git@github.com

# 4. 配置远程仓库（使用 SSH）
git remote set-url origin git@github.com:你的用户名/open-unlearning.git

# ========== 日常提交流程 ==========
# 1. 查看变更
git status
git diff

# 2. 暂存变更
git add .

# 3. 提交变更
git commit -m "你的提交信息"

# 4. 拉取远程更新（避免冲突）
git pull --rebase origin main

# 5. 推送到远程
git push

# 6. 验证
git status
```

---

## 五、快速参考命令

| 操作 | 命令 |
|------|------|
| 查看状态 | `git status` |
| 查看变更 | `git diff` |
| 暂存文件 | `git add .` |
| 提交变更 | `git commit -m "信息"` |
| 查看远程 | `git remote -v` |
| 修改远程地址 | `git remote set-url origin git@github.com:用户/仓库.git` |
| 拉取更新 | `git pull --rebase origin main` |
| 推送变更 | `git push` |
| 查看历史 | `git log --oneline` |
| 测试 SSH | `ssh -T git@github.com` |

---

## 六、最佳实践

1. **提交前先拉取**：`git pull --rebase` 避免冲突
2. **使用有意义的提交信息**：清晰描述做了什么改动
3. **小步提交**：每次提交一个逻辑完整的改动
4. **使用 SSH**：比 HTTPS 更方便和安全
5. **定期推送**：避免本地提交堆积过多
6. **查看状态**：推送前用 `git status` 确认状态

---

## 七、故障排查

### SSH 连接失败
```bash
# 测试连接
ssh -T git@github.com

# 如果失败，检查：
# 1. SSH 密钥是否添加到 GitHub
# 2. 密钥文件权限是否正确
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
```

### 认证问题
```bash
# 清除缓存的凭据（Linux）
git credential-cache exit

# 或使用 credential helper
git config --global credential.helper store
```

### 查看配置
```bash
git config --list              # 查看所有配置
git config user.name           # 查看用户名
git config user.email          # 查看邮箱
```

---

**提示：** 保存此文件作为参考，遇到问题时可以快速查找解决方案。
