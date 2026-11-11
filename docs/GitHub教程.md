# 把项目传到 GitHub（实际操作版）

> 你的仓库地址已经定好：**https://github.com/ningxling/Gomoku**
> 本地已经准备好一个 15 MB 的精简仓库（`五子棋项目\GitHub-发布包\`），只差"在能上网的地方推一次"。

---

## 第 0 步：先解决网络（这一步最容易卡）

GitHub 在**校园网/部分热点**下连不上。先在 PowerShell 里测一下：

```powershell
Test-NetConnection github.com -Port 443 -InformationLevel Quiet
```

- 返回 `True` → 可以继续；
- 返回 `False`（现在就是这种情况）→ 换网络（手机热点通常可以），或者挂学校的 VPN 再试。

---

## 第 1 步：在 GitHub 上建仓库（2 分钟）

1. 打开 <https://github.com/new>
2. **Repository name** 填 `Gomoku`
3. **不要**勾 "Add a README file"（否则会多一个提交，推送时要处理冲突）
4. 点 **Create repository**

如果你之前已经建过 `Gomoku` 这个仓库，就跳过这一步。

---

## 第 2 步：准备一个访问令牌（PAT）

GitHub 从 2021 年起**不接受账号密码**，推送时要用 "token" 当密码：

1. 打开 <https://github.com/settings/tokens>
2. **Generate new token** → **Generate new token (classic)**
3. Note 随便写（比如 `gomoku-push`），**Expiration** 选 30 天
4. 勾选 **`repo`**（整个大项）
5. 点最下面 **Generate token**，把以 `ghp_` 开头的字符串**复制下来**（只显示一次）

---

## 第 3 步：推送（1 分钟）

打开 PowerShell，粘这三行：

```powershell
cd "C:\Users\erich\Documents\deepseek-harness\default-workspace\五子棋项目\GitHub-发布包"
git push -u origin main
```

弹出的窗口里：

- **Username**：`ningxling`
- **Password**：粘贴刚才那个 `ghp_...` 令牌（**不是**账号密码）

看到 `branch 'main' set up to track 'origin/main'` 就成功了。

> 附带的 `推送.ps1` 就是这个命令的封装，右键"使用 PowerShell 运行"也行。

---

## 第 4 步：确认

打开 <https://github.com/ningxling/Gomoku>，应该能看到：

```
docs/          报告.pdf、技术说明、AI 协作记录
gomoku/        引擎、模型、MCTS、训练脚本
history/       训练曲线、手数分布、代际对战图、闸门日志
gomoku.html    网页
README.md
```

---

## 常见问题

**Q：提示 `remote: Permission to ... denied`**
A：令牌没勾 `repo`，或者用户名写错了。重新生成一个勾了 `repo` 的。

**Q：提示 `failed to connect` / `Connection timed out`**
A：网络到不了 GitHub，回到第 0 步。

**Q：提示 `Updates were rejected`**
A：远端仓库里已经有文件（比如建库时勾了 README）。两种解法：
- 删掉远端的 README（网页上直接删），再推；
- 或者强制覆盖（**会覆盖远端已有内容**）：`git push -u origin main --force`

**Q：以后改了代码怎么更新？**
A：

```powershell
cd "C:\Users\erich\Documents\deepseek-harness\default-workspace\五子棋项目\GitHub-发布包"
git add -A
git commit -m "说明这次改了什么"
git push
```

---

## 说明：为什么是"精简仓库"

完整仓库有 **594 MB**（含 250 份训练权重快照）。GitHub 虽然能收，但：

- 克隆一次要很久，评分老师大概率不会等；
- 超过 1 GB 会被警告甚至限流。

所以发布包只保留：

| 内容 | 大小 |
|---|---|
| 源码 + 网页 + 文档 + 图 | ~5 MB |
| 两份关键权重（2358 最终版、1997 加宽前） | ~9 MB |
| 训练日志与闸门记录（CSV） | ~1 MB |

其余权重（401 + 153 份快照）留在本地 `五子棋项目\模型快照-64ch\` 和 `模型快照-128ch\`，需要时可以单独发。
