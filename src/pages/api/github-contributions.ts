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

    // 解析 GitHub 贡献图：<td> 含 data-date、id、data-level（0–4）；<tool-tip> 给精确 count

    type Row = { count: number; level: number | null };
    const contributionsMap = new Map<string, Row>();
    let match;

    function ensureContributionEntry(date: string): Row {
      let row = contributionsMap.get(date);
      if (!row) {
        row = { count: 0, level: null };
        contributionsMap.set(date, row);
      }
      return row;
    }

    const idToDateMap = new Map<string, string>();

    const tdOpenRe = /<td\b([^>]*)>/g;
    while ((match = tdOpenRe.exec(htmlText)) !== null) {
      const attrs = match[1];
      const date = /data-date="([^"]+)"/.exec(attrs)?.[1];
      const id = /id="([^"]+)"/.exec(attrs)?.[1];
      if (!date || !id) continue;
      idToDateMap.set(id, date);
      const row = ensureContributionEntry(date);
      const levelRaw = /data-level="([^"]+)"/.exec(attrs)?.[1];
      if (levelRaw !== undefined && levelRaw !== '') {
        const lv = parseInt(levelRaw, 10);
        if (Number.isFinite(lv) && lv >= 0 && lv <= 4) row.level = lv;
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
          ensureContributionEntry(date).count = count;
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

        ensureContributionEntry(date).count = count;
      }
    }

    const contributions: { date: string; count: number; level?: number }[] = Array.from(
      contributionsMap.entries(),
    )
      .map(([date, row]) => {
        const o: { date: string; count: number; level?: number } = { date, count: row.count };
        if (row.level != null) o.level = row.level;
        return o;
      })
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
        
        ensureContributionEntry(date).count = count;
      }

      const fallbackContributions: { date: string; count: number; level?: number }[] = Array.from(
        contributionsMap.entries(),
      )
        .map(([date, row]) => {
          const o: { date: string; count: number; level?: number } = { date, count: row.count };
          if (row.level != null) o.level = row.level;
          return o;
        })
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