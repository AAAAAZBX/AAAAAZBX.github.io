# Waline 评论系统配置指南

## 第一步：部署 Waline 服务器

Waline 支持多种部署方式，推荐使用 **Vercel**（免费且简单）。

### 方式一：使用 Vercel 一键部署（推荐，最简单）

**方法 A：使用官方一键部署链接（推荐）**

1. **访问 Waline 官方部署页面**
   - 打开：https://waline.js.org/en/guide/deploy/vercel.html
   - 点击页面上的 "Deploy" 按钮
   - 这会自动跳转到 Vercel 并使用正确的模板

2. **或者直接访问 Vercel 模板**
   - 打开：https://vercel.com/templates/waline
   - 点击 "Deploy" 按钮

**方法 B：手动部署（如果方法 A 不可用）**

1. **删除当前项目（如果已创建）**
   - 在 Vercel 项目设置中删除当前失败的 waline 项目

2. **使用正确的仓库 URL**
   - 在 Vercel "New Project" 页面
   - 输入：`https://github.com/walinejs/waline`
   - 点击 "Continue"
   - **重要**：在配置页面，找到 "Root Directory" 设置
   - 输入：`packages/vercel`（注意是小写）
   - 如果找不到 Root Directory，在创建项目后，进入 Settings → Build and Deployment → Root Directory，设置为 `packages/vercel`

3. **配置 LeanCloud 数据库（必需）**
   - 访问：https://leancloud.app
   - 注册/登录账号
   - 创建新应用（Create app）
   - 进入应用 → Settings → App Keys
   - 记录以下信息：
     - `APP ID`
     - `APP Key`
     - `Master Key`

4. **在 Vercel 中添加环境变量**
   - 进入你的 Vercel 项目 → Settings → Environment Variables
   - 添加以下环境变量：
     - `LEAN_ID`: 你的 LeanCloud APP ID
     - `LEAN_KEY`: 你的 LeanCloud APP Key
     - `LEAN_MASTER_KEY`: 你的 LeanCloud Master Key
   - 添加完成后，进入 Deployments 页面，点击最新的部署，选择 "Redeploy"

5. **获取服务器 URL**
   - 部署完成后，Vercel 会提供一个 URL
   - 例如：`https://waline-xxxxx.vercel.app`
   - 这就是你的 Waline 服务器地址

6. **首次登录设置管理员**
   - 访问：`https://waline-xxxxx.vercel.app/ui/`
   - **重要**：第一个注册的账号会自动成为管理员
   - 务必在部署后立即登录并注册管理员账号

### 方式二：使用 Leancloud（需要注册账号）

参考官方文档：https://waline.js.org/guide/get-started.html

## 第二步：配置环境变量

### 本地开发环境

1. **创建 `.env` 文件**
   - 在项目根目录创建 `.env` 文件
   - 添加以下内容：
   ```env
   PUBLIC_WALINE_SERVER_URL=https://your-waline-app.vercel.app
   ```
   - 将 `https://your-waline-app.vercel.app` 替换为你的实际 Waline 服务器地址

2. **重启开发服务器**
   ```bash
   npm run dev
   ```

### 生产环境（GitHub Pages）

如果使用 GitHub Actions 部署，需要在 GitHub 仓库中配置 Secrets：

1. **进入 GitHub 仓库设置**
   - 打开你的 GitHub 仓库
   - 点击 `Settings` → `Secrets and variables` → `Actions`

2. **添加环境变量**
   - 点击 `New repository secret`
   - Name: `PUBLIC_WALINE_SERVER_URL`
   - Value: 你的 Waline 服务器地址（例如：`https://your-waline-app.vercel.app`）

3. **更新 GitHub Actions 工作流**
   
   编辑 `.github/workflows/deploy.yml`，在 build 步骤中添加环境变量：
   
   ```yaml
   - name: Install, build, and upload your site
     uses: withastro/action@v5
     env:
       PUBLIC_WALINE_SERVER_URL: ${{ secrets.PUBLIC_WALINE_SERVER_URL }}
   ```

## 第三步：验证配置

1. **访问任意博客文章页面**
2. **滚动到页面底部**
3. **应该能看到评论区域**
4. **如果显示 "Comment seems to stuck. Try to refresh?✨"**
   - 检查浏览器控制台是否有错误
   - 确认 `PUBLIC_WALINE_SERVER_URL` 配置正确
   - 确认 Waline 服务器可以正常访问

## 可选：配置邮件提醒

如果需要邮件提醒功能，可以在 Vercel 项目设置中添加环境变量：

1. **进入 Vercel 项目设置**
   - 打开你的 Vercel 项目
   - 点击 `Settings` → `Environment Variables`

2. **添加以下环境变量**
   ```
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=465
   SMTP_USER=your-email@gmail.com
   SMTP_PASS=your-app-password
   SMTP_SECURE=true
   SITE_NAME=你的网站名称
   SITE_URL=https://AAAAAZBX.github.io
   AUTHOR_EMAIL=your-email@gmail.com
   ```

   **注意**：如果使用 Gmail，需要：
   - 开启两步验证
   - 生成应用专用密码：https://myaccount.google.com/apppasswords

## 故障排查

### 评论不显示
- 检查 `PUBLIC_WALINE_SERVER_URL` 是否正确配置
- 检查 Waline 服务器是否可以访问
- 查看浏览器控制台的错误信息

### 无法登录/注册
- 确认已访问 `https://your-waline-app.vercel.app/ui/` 并注册管理员账号
- 检查 Waline 服务器日志

### 评论无法提交
- 检查网络连接
- 确认 Waline 服务器正常运行
- 查看浏览器控制台和服务器日志

## 参考链接

- Waline 官方文档：https://waline.js.org/
- Vercel 部署指南：https://waline.js.org/guide/deploy/vercel.html
- 邮件配置指南：https://waline.js.org/guide/features/notification.html
