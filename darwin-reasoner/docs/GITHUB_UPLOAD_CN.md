# GitHub 上传教程（推荐 Private Repo + SSH）

## A. GitHub 网页端

1. 登录 GitHub。
2. 右上角 `+` -> `New repository`。
3. Repository name 填 `darwin-reasoner`。
4. 选择 **Private**。
5. 不要勾选自动创建 README / .gitignore / license，因为压缩包里已经有。
6. 创建 repository。

## B. 本地电脑第一次上传

解压本文件包，进入目录：

```bash
cd darwin-reasoner
```

检查秘密没有被放进去：

```bash
git status
find . -maxdepth 2 -name '.env' -o -name '*.pem' -o -name '*.key'
```

第一次使用 Git 时配置身份：

```bash
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

### 推荐：配置 SSH

```bash
ssh-keygen -t ed25519 -C "YOUR_GITHUB_EMAIL"
cat ~/.ssh/id_ed25519.pub
```

复制输出的**公钥**到 GitHub：Settings -> SSH and GPG keys -> New SSH key。

绝对不要上传 `~/.ssh/id_ed25519` 私钥。

测试：

```bash
ssh -T git@github.com
```

### 初始化并推送

```bash
git init
git branch -M main
git add .
git status
git commit -m "init: DarwinReasoner research harness"
git remote add origin git@github.com:YOUR_USERNAME/darwin-reasoner.git
git push -u origin main
```

以后每次修改：

```bash
git add .
git commit -m "feat: describe your change"
git push
```

## C. 把借卡的人加进 Private Repo

GitHub repository -> Settings -> Collaborators / Manage access -> Add people。

最好让对方使用自己的 GitHub 账号，不要共享你的 token。

## D. A100 服务器 clone

在服务器上单独生成 SSH key：

```bash
ssh-keygen -t ed25519 -C "a100-server"
cat ~/.ssh/id_ed25519.pub
```

将服务器公钥添加到有仓库访问权限的 GitHub 账号，或者使用 repo deploy key（是否允许写入按你的协作需求决定）。

然后：

```bash
git clone git@github.com:YOUR_USERNAME/darwin-reasoner.git
cd darwin-reasoner
```

## E. 推荐分支规则

- `main`：确认能跑的版本；
- `dev`：日常开发；
- `exp/<name>`：某次危险或大型实验修改。

例如：

```bash
git checkout -b dev
git push -u origin dev
```

服务器正式实验尽量 checkout 一个明确 commit：

```bash
git checkout main
git pull
git rev-parse HEAD
```

每个 run 会自动记录当前 commit hash。

## F. 上传前 30 秒检查

```bash
git status
git diff --cached
```

确认没有 `.env`、token、checkpoint、模型权重、dataset cache、rollouts 后再 push。
