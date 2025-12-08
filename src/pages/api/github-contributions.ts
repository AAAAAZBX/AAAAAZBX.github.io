import type { APIRoute } from 'astro';

export const GET: APIRoute = async () => {
  const username = 'AAAAAZBX';
  
  try {
    // 从 GitHub 获取贡献图 HTML
    const githubUrl = `https://github.com/users/${username}/contributions`;
    const response = await fetch(githubUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://github.com/',
      },
    });
    
    if (!response.ok) {
      throw new Error(`GitHub returned ${response.status}: ${response.statusText}`);
    }
    
    const htmlText = await response.text();
    if (!htmlText || htmlText.length === 0) {
      throw new Error('Empty response from GitHub');
    }
    
    // GitHub 的贡献图现在使用 <td> 元素，包含 data-date 和 data-level 属性
    const contributions: { date: string; count: number }[] = [];
    
    // 匹配所有包含 data-date 的 <td> 元素
    const tdPattern = /<td[^>]*data-date="([^"]+)"[^>]*data-level="(\d+)"[^>]*>/g;
    let match;
    
    while ((match = tdPattern.exec(htmlText)) !== null) {
      const date = match[1];
      const level = parseInt(match[2], 10);
      
      // GitHub 的 level 映射：0=无, 1=1-3次, 2=4-6次, 3=7-9次, 4=10+次
      // 使用最小值的近似
      let count = 0;
      if (level === 1) count = 1;
      else if (level === 2) count = 4;
      else if (level === 3) count = 7;
      else if (level === 4) count = 10;
      
      contributions.push({ date, count });
    }
    
    // 如果没找到，尝试另一种模式（没有 data-level 的）
    if (contributions.length === 0) {
      const tdPattern2 = /<td[^>]*data-date="([^"]+)"[^>]*>/g;
      while ((match = tdPattern2.exec(htmlText)) !== null) {
        const date = match[1];
        contributions.push({ date, count: 0 });
      }
    }
    
    if (contributions.length === 0) {
      throw new Error('No contribution data found in HTML');
    }
    
    const totalContributions = contributions.reduce((sum, c) => sum + c.count, 0);
    
    return new Response(JSON.stringify({ 
      contributions,
      totalContributions
    }), {
      status: 200,
      headers: { 
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=3600',
      },
    });
  } catch (error: any) {
    console.error('Error fetching GitHub contributions:', error);
    return new Response(JSON.stringify({ 
      contributions: [],
      error: error?.message || 'Unknown error',
      totalContributions: 0
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
};
