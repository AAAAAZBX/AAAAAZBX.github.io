import type { APIRoute } from 'astro';

const USERNAME = 'AAAAAZBX';

// GitHub GraphQL 返回的颜色 → level (0–4) 映射
const COLOR_TO_LEVEL: Record<string, number> = {
  '#ebedf0': 0,
  '#9be9a8': 1,
  '#40c463': 2,
  '#30a14e': 3,
  '#216e39': 4,
};

/**
 * 通过 GitHub GraphQL API 获取贡献数据（含私有贡献，需认证）。
 * 返回 null 表示失败，调用方应回退到 HTML 抓取。
 */
async function fetchViaGraphQL(): Promise<{
  contributions: { date: string; count: number; level: number }[];
  totalContributions: number;
} | null> {
  const token = import.meta.env.GITHUB_TOKEN;
  if (!token) {
    console.log('[API] No GITHUB_TOKEN, skipping GraphQL API');
    return null;
  }

  const query = `
    query($username: String!) {
      user(login: $username) {
        contributionsCollection {
          contributionCalendar {
            totalContributions
            weeks {
              contributionDays {
                date
                contributionCount
                color
              }
            }
          }
        }
      }
    }
  `;

  try {
    console.log('[API] Fetching contributions via GitHub GraphQL API...');
    const response = await fetch('https://api.github.com/graphql', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
        'User-Agent': 'tremendous-matter-blog',
      },
      body: JSON.stringify({ query, variables: { username: USERNAME } }),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new Error(`GraphQL API returned ${response.status}: ${body.slice(0, 200)}`);
    }

    const json = await response.json();
    if (json.errors) {
      throw new Error(`GraphQL errors: ${json.errors.map((e: any) => e.message).join('; ')}`);
    }

    const calendar = json.data?.user?.contributionsCollection?.contributionCalendar;
    if (!calendar) {
      throw new Error('Unexpected GraphQL response structure');
    }

    const contributions: { date: string; count: number; level: number }[] = [];
    for (const week of calendar.weeks) {
      for (const day of week.contributionDays) {
        contributions.push({
          date: day.date,
          count: day.contributionCount,
          level: COLOR_TO_LEVEL[day.color] ?? 0,
        });
      }
    }

    contributions.sort((a, b) => a.date.localeCompare(b.date));

    console.log(`[API] GraphQL: ${contributions.length} days, total: ${calendar.totalContributions}`);
    return { contributions, totalContributions: calendar.totalContributions };
  } catch (error: any) {
    console.error('[API] GraphQL failed:', error.message);
    return null;
  }
}

/**
 * 回退方案：从 GitHub 公开贡献页面抓取 HTML 并解析。
 */
async function fetchViaHtmlScraping(): Promise<{
  contributions: { date: string; count: number; level?: number }[];
  totalContributions: number;
}> {
  const githubUrl = `https://github.com/users/${USERNAME}/contributions`;

  console.log('[API] Falling back to HTML scraping...');

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

  const tooltipPattern = /<tool-tip[^>]*for="([^"]+)"[^>]*>([^<]*)<\/tool-tip>/g;
  while ((match = tooltipPattern.exec(htmlText)) !== null) {
    const forId = match[1];
    const tooltipText = match[2];
    const contributionMatch = tooltipText.match(/(\d+)\s+contribution/i);
    if (contributionMatch) {
      const count = parseInt(contributionMatch[1], 10) || 0;
      const date = idToDateMap.get(forId);
      if (date) {
        ensureContributionEntry(date).count = count;
      }
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

  if (contributions.length === 0) {
    throw new Error('No contribution data found in HTML');
  }

  const totalContributions = contributions.reduce((sum, c) => sum + c.count, 0);
  console.log(`[API] HTML scraping: ${contributions.length} days, total: ${totalContributions}`);

  return { contributions, totalContributions };
}

export const GET: APIRoute = async () => {
  try {
    // 优先使用 GraphQL API（含私有贡献）
    const graphqlResult = await fetchViaGraphQL();
    if (graphqlResult && graphqlResult.contributions.length > 0) {
      return new Response(JSON.stringify(graphqlResult), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=3600',
        },
      });
    }

    // 回退到 HTML 抓取
    const htmlResult = await fetchViaHtmlScraping();
    return new Response(JSON.stringify(htmlResult), {
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
      totalContributions: 0,
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
};