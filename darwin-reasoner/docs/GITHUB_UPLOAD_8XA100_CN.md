# GitHub 上传与服务器协作

## 1. 创建仓库

GitHub -> New repository：

- Repository name: `darwin-reasoner`
- Visibility: **Private**
- 不要额外初始化 README / .gitignore / License

## 2. Windows 本地上传

解压 ZIP，进入 `darwin-reasoner` 文件夹，在 PowerShell/Git Bash 执行：

```bash
git init
git branch -M main
git add .
git status
git commit -m "init: DarwinReasoner 8xA100 research harness"
git remote add origin git@github.com:YOUR_NAME/darwin-reasoner.git
git push -u origin main
```

如果你还没有 GitHub SSH key：

```bash
ssh-keygen -t ed25519 -C "YOUR_GITHUB_EMAIL"
```

把 `~/.ssh/id_ed25519.pub` 的**公钥**添加到 GitHub -> Settings -> SSH and GPG keys。不要上传私钥。

## 3. 添加服务器协作者

仓库 -> Settings -> Collaborators，邀请借你 A100 的协作者。

服务器自己也应生成 SSH key，然后：

```bash
git clone git@github.com:YOUR_NAME/darwin-reasoner.git
cd darwin-reasoner
```

## 4. 服务器第一次部署

```bash
bash scripts/setup_a100.sh
source .venv/bin/activate
bash scripts/check_8xa100.sh
bash scripts/smoke.sh
```

如果需要 Hugging Face token，服务器本地登录即可：

```bash
hf auth login
```

不要把 token 写入 GitHub。

## 5. 真正开始实验

先准备公开数据：

```bash
python scripts/bootstrap_public_data.py
```

先跑 pilot counterfactual collection，再构建 policy dataset，最后：

```bash
bash scripts/train_policy_8xa100.sh
```

## 6. 日常协作

你修改代码：

```bash
git add .
git commit -m "feat: describe change"
git push
```

服务器运行前：

```bash
git pull
```

大模型、数据缓存、checkpoint、rollout 不要上传 GitHub。`.gitignore` 已默认忽略这些目录。
