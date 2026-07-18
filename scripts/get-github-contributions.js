// 构建时获取GitHub贡献数据的脚本
// 优先使用 GitHub GraphQL API（认证后可获取私有贡献），失败时回退到 HTML 抓取
import { writeFileSync, mkdirSync, readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 手动加载 .env（避免额外依赖 dotenv）
try {
  const envPath = join(__dirname, '..', '.env');
  const envContent = readFileSync(envPath, 'utf-8');
  for (const line of envContent.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx === -1) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    const value = trimmed.slice(eqIdx + 1).trim();
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
  console.log('[Build] Loaded .env file');
} catch (e) {
  // .env 不存在就跳过
}

const USERNAME = 'AAAAAZBX';

// GitHub GraphQL 返回的颜色 → level (0–4) 映射
const COLOR_TO_LEVEL = {
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
async function fetchViaGraphQL() {
  const token = process.env.GITHUB_TOKEN;
  if (!token) {
    console.log('[Build] No GITHUB_TOKEN env var, skipping GraphQL API');
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
    console.log('[Build] Fetching contributions via GitHub GraphQL API (includes private contributions)...');

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
      throw new Error(`GraphQL errors: ${json.errors.map(e => e.message).join('; ')}`);
    }

    const calendar = json.data?.user?.contributionsCollection?.contributionCalendar;
    if (!calendar) {
      throw new Error('Unexpected GraphQL response structure');
    }

    const contributions = [];
    for (const week of calendar.weeks) {
      for (const day of week.contributionDays) {
        const level = COLOR_TO_LEVEL[day.color] ?? 0;
        contributions.push({
          date: day.date,
          count: day.contributionCount,
          level,
        });
      }
    }

    contributions.sort((a, b) => a.date.localeCompare(b.date));

    const totalContributions = calendar.totalContributions;
    console.log(`[Build] GraphQL: ${contributions.length} days, total: ${totalContributions} (includes private)`);

    return { contributions, totalContributions };
  } catch (error) {
    console.error('[Build] GraphQL API failed:', error.message);
    return null;
  }
}

/**
 * 回退方案：从 GitHub 公开贡献页面抓取 HTML 并解析。
 * 注意：公开页面不含私有贡献。
 */
async function fetchViaHtmlScraping() {
  const githubUrl = `https://github.com/users/${USERNAME}/contributions`;

  console.log('[Build] Falling back to HTML scraping (public contributions only)...');

  const response = await fetch(githubUrl, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    },
  });

  if (!response.ok) {
    throw new Error(`GitHub returned ${response.status} ${response.statusText}`);
  }

  const htmlText = await response.text();

  if (!htmlText || htmlText.length === 0) {
    throw new Error('Empty response from GitHub');
  }

  /** @type {Map<string, { count: number, level: number | null }>} */
  const contributionsMap = new Map();
  let match;

  function ensureContributionEntry(date) {
    if (!contributionsMap.has(date)) {
      contributionsMap.set(date, { count: 0, level: null });
    }
    return contributionsMap.get(date);
  }

  const idToDateMap = new Map();

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

  console.log('[Build] Found', idToDateMap.size, 'td elements with data-date');

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

  if (contributionsMap.size === 0) {
    console.log('[Build] No td elements found, trying rect format...');
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

  const contributions = Array.from(contributionsMap.entries())
    .map(([date, row]) => {
      const o = { date, count: row.count };
      if (row.level != null) o.level = row.level;
      return o;
    })
    .sort((a, b) => a.date.localeCompare(b.date));

  if (contributions.length === 0) {
    console.error('[Build] No contributions found in HTML');
    return { contributions: [], totalContributions: 0 };
  }

  const totalContributions = contributions.reduce((sum, c) => sum + c.count, 0);
  console.log(`[Build] HTML scraping: ${contributions.length} days, total: ${totalContributions} (public only)`);

  return { contributions, totalContributions };
}

async function fetchGitHubContributions() {
  try {
    // 优先使用 GraphQL API（含私有贡献）
    const graphqlResult = await fetchViaGraphQL();
    if (graphqlResult && graphqlResult.contributions.length > 0) {
      return graphqlResult;
    }

    // 回退到 HTML 抓取
    return await fetchViaHtmlScraping();
  } catch (error) {
    console.error('[Build] Error fetching GitHub contributions:', error);
    return { contributions: [], totalContributions: 0, error: `Unable to fetch contributions data (${error.message})` };
  }
}

async function main() {
  try {
    const { contributions, totalContributions, error } = await fetchGitHubContributions();

    // 保存到 public 目录
    const outputPath = join(__dirname, '..', 'public', 'data', 'github-contributions.json');
    const outputDir = dirname(outputPath);
    mkdirSync(outputDir, { recursive: true });

    const data = {
      contributions,
      totalContributions,
      error,
      timestamp: new Date().toISOString(),
    };

    writeFileSync(outputPath, JSON.stringify(data, null, 2), 'utf-8');
    console.log(`[Build] Saved GitHub contributions to ${outputPath}`);

  } catch (error) {
    console.error('[Build] Error:', error);
    process.exit(1);
  }
}

main();
