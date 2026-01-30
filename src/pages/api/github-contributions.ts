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
    // GitHub 2024+ 新结构：
    // - <td> 元素包含 data-date 和 id
    // - <tool-tip> 元素包含精确贡献数，通过 for 属性关联到 td 的 id

    const contributionsMap = new Map<string, number>();
    let match;

    // 步骤1: 解析所有 <td> 元素，建立 id -> date 的映射
    const idToDateMap = new Map<string, string>();
    const tdPattern = /<td[^>]*data-date="([^"]+)"[^>]*id="([^"]+)"[^>]*>/g;
    
    while ((match = tdPattern.exec(htmlText)) !== null) {
      const date = match[1];
      const id = match[2];
      idToDateMap.set(id, date);
      // 初始化贡献数为 0
      if (!contributionsMap.has(date)) {
        contributionsMap.set(date, 0);
      }
    }

    // 也尝试 id 在 data-date 之前的情况
    const tdPatternAlt = /<td[^>]*id="([^"]+)"[^>]*data-date="([^"]+)"[^>]*>/g;
    while ((match = tdPatternAlt.exec(htmlText)) !== null) {
      const id = match[1];
      const date = match[2];
      if (!idToDateMap.has(id)) {
        idToDateMap.set(id, date);
        if (!contributionsMap.has(date)) {
          contributionsMap.set(date, 0);
        }
      }
    }

    console.log('[API] Found', idToDateMap.size, 'td elements with data-date');

    // 步骤2: 解析所有 <tool-tip> 元素，提取贡献数并关联到日期
    const tooltipPattern = /<tool-tip[^>]*for="([^"]+)"[^>]*>([^<]*)<\/tool-tip>/g;
    
    while ((match = tooltipPattern.exec(htmlText)) !== null) {
      const forId = match[1];
      const tooltipText = match[2];
      
      // 从 tooltip 文本中提取贡献数
      const contributionMatch = tooltipText.match(/(\d+)\s+contribution/i);
      if (contributionMatch) {
        const count = parseInt(contributionMatch[1], 10) || 0;
        
        // 通过 id 找到对应的日期
        const date = idToDateMap.get(forId);
        if (date) {
          contributionsMap.set(date, count);
        }
      }
    }

    // 如果新格式没有匹配到数据，尝试旧的 rect 格式作为备选
    if (contributionsMap.size === 0) {
      console.log('[API] No td elements found, trying rect format...');
      
      const rectPattern = /<rect[^>]*data-date="([^"]+)"[^>]*>([\s\S]*?)<\/rect>/g;
      while ((match = rectPattern.exec(htmlText)) !== null) {
        const date = match[1];
        const innerContent = match[2];
        
        if (contributionsMap.has(date)) continue;
        
        let count = 0;
        const titleMatch = innerContent.match(/<title>([^<]*)<\/title>/);
        if (titleMatch) {
          const contributionMatch = titleMatch[1].match(/(\d+)\s+contribution/i);
          if (contributionMatch) {
            count = parseInt(contributionMatch[1], 10) || 0;
          }
        }
        
        contributionsMap.set(date, count);
      }
    }

    // 转换为数组并按日期排序
    const contributions: { date: string; count: number }[] = Array.from(contributionsMap.entries())
      .map(([date, count]) => ({ date, count }))
      .sort((a, b) => a.date.localeCompare(b.date));

    // 如果没有找到数据，尝试旧的 table 格式作为备选
    if (contributions.length === 0) {
      // 匹配 td 元素，尝试从 title 属性获取精确贡献数
      const tdPattern = /<td[^>]*data-date="([^"]+)"[^>]*>/g;
      while ((match = tdPattern.exec(htmlText)) !== null) {
        const tdTag = match[0];
        const date = match[1];

        if (contributionsMap.has(date)) continue;

        let count = 0;
        
        // 尝试从 title 属性获取精确数值
        const titleAttrMatch = tdTag.match(/title="([^"]*)"/);
        if (titleAttrMatch) {
          const titleStr = titleAttrMatch[1];
          const contributionMatch = titleStr.match(/(\d+)\s+contribution/i);
          if (contributionMatch) {
            count = parseInt(contributionMatch[1], 10) || 0;
          }
        }
        
        // 注意：不再使用 data-level 估算

        contributionsMap.set(date, count);
      }

      // 重新生成排序后的数组
      const fallbackContributions: { date: string; count: number }[] = Array.from(contributionsMap.entries())
        .map(([date, count]) => ({ date, count }))
        .sort((a, b) => a.date.localeCompare(b.date));

      if (fallbackContributions.length > 0) {
        contributions.length = 0;
        contributions.push(...fallbackContributions);
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