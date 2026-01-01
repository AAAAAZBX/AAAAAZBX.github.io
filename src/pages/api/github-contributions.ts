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
    
    // 调试：检查 HTML 中是否包含 SVG 或 table
    const hasSvg = htmlText.includes('<svg');
    const hasTable = htmlText.includes('<table');
    const hasRect = htmlText.includes('<rect');
    const hasTd = htmlText.includes('<td');
    console.log('[API] HTML check:', { hasSvg, hasTable, hasRect, hasTd, htmlLength: htmlText.length });
    
    // 解析 GitHub 贡献图数据
    // GitHub 现在使用 SVG 格式，每个 <rect> 元素包含：
    // - data-date: 日期
    // - data-level: 贡献等级（0-4）
    // - data-count: 精确贡献数（如果存在）
    // - <title>: 包含精确贡献数的 tooltip（如 "6 contributions on 2025-11-22"）
    const contributions: { date: string; count: number }[] = [];
    const seenDates = new Set<string>();
    
    // 方法1: 匹配 SVG 中的 <rect> 元素及其内容（包括 <title> 元素）
    // GitHub 的格式可能是：
    // <rect data-date="2024-01-01" data-level="3" ...><title>5 contributions on Mon, Jan 1, 2024</title></rect>
    // 或者自闭合的：<rect data-date="2024-01-01" data-level="3" ... />
    
    // 首先尝试匹配包含 <title> 的完整 rect 元素（非贪婪匹配，避免匹配到下一个 rect）
    const rectWithTitlePattern = /<rect[^>]*data-date="([^"]+)"[^>]*>[\s\S]*?<\/rect>/g;
    let match;
    
    while ((match = rectWithTitlePattern.exec(htmlText)) !== null) {
      const rectBlock = match[0];
      const date = match[1];
      
      if (seenDates.has(date)) continue;
      seenDates.add(date);
      
      let count = 0;
      let extractedFromTitle = false;
      
      // 优先级1: data-count（最精确，如果存在）
      const countMatch = rectBlock.match(/data-count="(\d+)"/);
      if (countMatch) {
        count = parseInt(countMatch[1], 10) || 0;
      } else {
        // 优先级2: 从 <title> 元素中提取（最可靠的方法）
        const titleMatch = rectBlock.match(/<title>([^<]*)<\/title>/);
        if (titleMatch) {
          const titleText = titleMatch[1];
          // 匹配 "X contributions on ..." 或 "X contribution on ..."（单数）
          const contributionMatch = titleText.match(/(\d+)\s+contribution/i);
          if (contributionMatch) {
            count = parseInt(contributionMatch[1], 10) || 0;
            extractedFromTitle = true;
          } else if (titleText.toLowerCase().includes('no contribution')) {
            count = 0;
            extractedFromTitle = true;
          }
        }
        
        // 优先级3: 从 title 属性中提取
        if (!extractedFromTitle) {
          const titleAttrMatch = rectBlock.match(/title="([^"]*)"/);
          if (titleAttrMatch) {
            const titleStr = titleAttrMatch[1];
            const contributionMatch = titleStr.match(/(\d+)\s+contribution/i);
            if (contributionMatch) {
              count = parseInt(contributionMatch[1], 10) || 0;
              extractedFromTitle = true;
            } else if (titleStr.toLowerCase().includes('no contribution')) {
              count = 0;
              extractedFromTitle = true;
            }
          }
        }
        
        // 优先级4: 使用 data-level 转换（最后的手段，不准确）
        if (!extractedFromTitle) {
          const levelMatch = rectBlock.match(/data-level="(\d+)"/);
          if (levelMatch) {
            const level = parseInt(levelMatch[1], 10);
            if (level === 1) count = 1;
            else if (level === 2) count = 3; // 2-3 的平均值向上取整
            else if (level === 3) count = 5;  // 4-5 的平均值向上取整
            else if (level === 4) count = 6;  // 6+ 的最小值
          }
        }
      }
      
      contributions.push({ date, count });
    }
    
    // 方法1.5: 如果上面的方法没找到，尝试匹配自闭合的 rect 元素
    if (contributions.length === 0) {
      const rectSelfClosingPattern = /<rect[^>]*data-date="([^"]+)"[^>]*\/>/g;
      while ((match = rectSelfClosingPattern.exec(htmlText)) !== null) {
        const rectBlock = match[0];
        const date = match[1];
        
        if (seenDates.has(date)) continue;
        seenDates.add(date);
        
        let count = 0;
        
        // 提取 data-count
        const countMatch = rectBlock.match(/data-count="(\d+)"/);
        if (countMatch) {
          count = parseInt(countMatch[1], 10) || 0;
        } else {
          // 提取 title 属性
          const titleMatch = rectBlock.match(/title="([^"]*)"/);
          if (titleMatch) {
            const titleStr = titleMatch[1];
            const contributionMatch = titleStr.match(/(\d+)\s+contribution/i);
            if (contributionMatch) {
              count = parseInt(contributionMatch[1], 10) || 0;
            } else if (titleStr.toLowerCase().includes('no contribution')) {
              count = 0;
            }
          }
          
          // 使用 data-level
          if (count === 0) {
            const levelMatch = rectBlock.match(/data-level="(\d+)"/);
            if (levelMatch) {
              const level = parseInt(levelMatch[1], 10);
              if (level === 1) count = 1;
              else if (level === 2) count = 3;
              else if (level === 3) count = 5;
              else if (level === 4) count = 6;
            }
          }
        }
        
        contributions.push({ date, count });
      }
    }
    
    // 方法2: 如果 SVG 方法没找到，尝试匹配 <td> 元素（GitHub 旧格式）
    if (contributions.length === 0) {
      const tdPattern = /<td[^>]*data-date="([^"]+)"[^>]*data-level="(\d+)"[^>]*>/g;
      while ((match = tdPattern.exec(htmlText)) !== null) {
        const date = match[1];
        const level = parseInt(match[2], 10);
        
        if (seenDates.has(date)) continue;
        seenDates.add(date);
        
        let count = 0;
        if (level === 1) count = 1;
        else if (level === 2) count = 4;
        else if (level === 3) count = 7;
        else if (level === 4) count = 10;
        
        contributions.push({ date, count });
      }
    }
    
    if (contributions.length === 0) {
      // 输出部分 HTML 用于调试
      const svgMatch = htmlText.match(/<svg[^>]*>[\s\S]{0,5000}/);
      const tableMatch = htmlText.match(/<table[^>]*>[\s\S]{0,5000}/);
      const rectMatch = htmlText.match(/<rect[^>]*data-date[^>]*>[\s\S]{0,1000}/);
      console.error('[API] No contributions found. HTML length:', htmlText.length);
      console.error('[API] SVG snippet:', svgMatch ? svgMatch[0].substring(0, 500) : 'No SVG');
      console.error('[API] Table snippet:', tableMatch ? tableMatch[0].substring(0, 500) : 'No table');
      console.error('[API] Rect snippet:', rectMatch ? rectMatch[0].substring(0, 500) : 'No rect');
      
      // 尝试查找所有包含 data-date 的内容
      const allDataDates = htmlText.match(/data-date="([^"]+)"/g);
      console.error('[API] Found data-date attributes:', allDataDates ? allDataDates.length : 0);
      if (allDataDates && allDataDates.length > 0) {
        console.error('[API] First 5 data-date values:', allDataDates.slice(0, 5));
      }
      
      throw new Error('No contribution data found in HTML');
    }
    
    const totalContributions = contributions.reduce((sum, c) => sum + c.count, 0);
    console.log(`[API] Parsed ${contributions.length} contribution days, total: ${totalContributions}`);
    
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
