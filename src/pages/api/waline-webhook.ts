import type { APIRoute } from 'astro';

/**
 * Waline Webhook 处理路由
 * 用于接收 Waline 评论事件并发送邮件通知
 * 
 * 注意：此功能需要在 Waline 服务端配置 webhook URL
 * Webhook URL: https://aaaaazbx.github.io/api/waline-webhook
 * 
 * 如果使用 GitHub Pages，此功能可能无法正常工作（GitHub Pages 不支持 serverless functions）
 * 建议使用方案一：直接在 Waline 服务端配置邮件通知
 */

interface WalineWebhookEvent {
  type: 'new' | 'update' | 'delete' | 'spam';
  comment: {
    id: number;
    author: string;
    mail: string;
    url?: string;
    ip: string;
    ua: string;
    comment: string;
    pid?: number;
    rid?: number;
    link: string;
    status: 'approved' | 'waiting' | 'spam';
    created: number;
    updated: number;
  };
  page: {
    id: number;
    title: string;
    url: string;
  };
}

export const POST: APIRoute = async ({ request }) => {
  try {
    // 验证请求来源（可选，增加安全性）
    const authHeader = request.headers.get('authorization');
    const expectedToken = import.meta.env.WALINE_WEBHOOK_TOKEN;
    
    if (expectedToken && authHeader !== `Bearer ${expectedToken}`) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const event: WalineWebhookEvent = await request.json();
    
    // 只处理新评论事件
    if (event.type !== 'new') {
      return new Response(JSON.stringify({ message: 'Event type not handled' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // 获取配置
    const authorEmail = import.meta.env.AUTHOR_EMAIL || 'zbx@stu.njau.edu.cn';
    const siteName = import.meta.env.SITE_NAME || 'Boxuan Zhang 的个人博客';
    const siteUrl = import.meta.env.SITE_URL || 'https://aaaaazbx.github.io';

    // 构建邮件内容
    const commentAuthor = event.comment.author;
    const commentContent = event.comment.comment;
    const commentUrl = `${siteUrl}${event.page.url}#comment-${event.comment.id}`;
    const pageTitle = event.page.title;

    const emailSubject = `[${siteName}] 新评论：${pageTitle}`;
    const emailBody = `
您收到了一条新评论：

评论者：${commentAuthor}
邮箱：${event.comment.mail}
页面：${pageTitle}
链接：${commentUrl}

评论内容：
${commentContent}

---
此邮件由 ${siteName} 自动发送
`;

    // 发送邮件（需要配置邮件服务）
    // 这里使用 nodemailer 或其他邮件服务
    // 注意：GitHub Pages 不支持 serverless functions，此功能需要部署到支持 serverless 的平台
    
    // 示例：使用 SendGrid、Resend、或其他邮件服务
    // 由于 GitHub Pages 限制，建议使用方案一：在 Waline 服务端直接配置邮件通知

    console.log('New comment received:', {
      author: commentAuthor,
      page: pageTitle,
      url: commentUrl,
    });

    // 返回成功响应
    return new Response(JSON.stringify({ 
      message: 'Webhook received successfully',
      commentId: event.comment.id
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });

  } catch (error: any) {
    console.error('Error processing Waline webhook:', error);
    return new Response(JSON.stringify({ 
      error: error?.message || 'Internal server error'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
};

// 支持 GET 请求用于测试
export const GET: APIRoute = async () => {
  return new Response(JSON.stringify({ 
    message: 'Waline webhook endpoint is active',
    note: 'This endpoint receives POST requests from Waline server',
    recommendation: 'For GitHub Pages deployment, use Waline server-side email notification instead'
  }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
};
