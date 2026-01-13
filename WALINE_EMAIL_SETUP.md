# Waline 邮件通知配置指南

## 方案一：在 Waline 服务端配置（推荐）

Waline 本身支持邮件通知功能，需要在 Waline 服务端（通常是部署在 Vercel 或其他平台）配置环境变量。

### 配置步骤

1. **获取 Gmail 应用密码**（如果使用 Gmail）
   - 前往 [Google 应用密码](https://myaccount.google.com/apppasswords)
   - 如果没有开启 2FA，需要先开启
   - 生成一个应用密码

2. **在 Waline 服务端配置环境变量**

   在 Vercel 项目设置中添加以下环境变量：

   ```env
   # SMTP 邮件服务配置
   SMTP_USER=your-gmail-account@gmail.com
   SMTP_PASS=your-gmail-app-password
   SMTP_SERVICE=Gmail
   SMTP_SECURE=TRUE
   
   # 网站信息
   SITE_NAME=你的网站名称
   SITE_URL=https://aaaaazbx.github.io
   AUTHOR_EMAIL=zbx@stu.njau.edu.cn
   ```

3. **其他邮件服务商配置**

   **QQ 邮箱**：
   ```env
   SMTP_USER=your-qq@qq.com
   SMTP_PASS=your-qq-email-auth-code
   SMTP_SERVICE=QQ
   SMTP_HOST=smtp.qq.com
   SMTP_PORT=587
   SMTP_SECURE=FALSE
   ```

   **163 邮箱**：
   ```env
   SMTP_USER=your-email@163.com
   SMTP_PASS=your-163-email-auth-code
   SMTP_SERVICE=163
   SMTP_HOST=smtp.163.com
   SMTP_PORT=465
   SMTP_SECURE=TRUE
   ```

4. **重新部署 Waline 服务端**

   配置完成后，重新部署 Waline 服务端，邮件通知功能即可生效。

### 参考文档

- [Waline 邮件通知官方文档](https://waline.js.org/guide/features/notification.html#%E9%82%AE%E4%BB%B6%E9%80%9A%E7%9F%A5)

---

## 方案二：通过 Astro API 路由实现（自定义方案）

如果你想使用自定义的邮件通知逻辑，可以通过 Waline 的 webhook 功能配合 Astro API 路由实现。

### 实现步骤

1. **配置 Waline Webhook**
   - 在 Waline 服务端配置 webhook URL：`https://aaaaazbx.github.io/api/waline-webhook`

2. **创建 API 路由**
   - 已创建 `src/pages/api/waline-webhook.ts`
   - 该路由会接收 Waline 的评论事件并发送邮件

3. **配置邮件服务环境变量**
   - 在 Astro 项目或部署平台配置邮件服务相关环境变量

### 注意事项

- **方案二限制**：如果使用 GitHub Pages 部署，API 路由无法正常工作（GitHub Pages 不支持 serverless functions）
- **推荐方案**：使用方案一，直接在 Waline 服务端配置邮件通知，这是最简单可靠的方式
- 如果使用 Vercel、Netlify 等支持 serverless 的平台，可以使用方案二

---

## 快速配置检查清单

- [ ] 确认 Waline 服务端已部署（Vercel/其他平台）
- [ ] 获取邮件服务商的应用密码（Gmail/QQ/163等）
- [ ] 在 Waline 服务端配置环境变量（SMTP相关）
- [ ] 配置网站信息（SITE_NAME, SITE_URL, AUTHOR_EMAIL）
- [ ] 重新部署 Waline 服务端
- [ ] 测试评论功能，确认收到邮件通知

## 常见问题

### Q: 为什么收不到邮件？
A: 检查以下几点：
1. 确认环境变量配置正确
2. 确认邮件服务商的应用密码正确
3. 检查垃圾邮件文件夹
4. 查看 Waline 服务端日志

### Q: GitHub Pages 可以使用方案二吗？
A: 不可以。GitHub Pages 是静态网站托管，不支持 serverless functions。请使用方案一。

### Q: 可以使用其他邮件服务商吗？
A: 可以。Waline 支持多种邮件服务商，包括 Gmail、QQ、163、Outlook 等。参考 [Waline 官方文档](https://waline.js.org/guide/features/notification.html) 进行配置。
