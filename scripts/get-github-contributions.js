// 构建时获取GitHub贡献数据的脚本
import { execSync } from 'child_process';
import { writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function fetchGitHubContributions() {
  const username = 'AAAAAZBX';
  const githubUrl = `https://github.com/users/${username}/contributions`;

  try {
    console.log('[Build] Fetching GitHub contributions...');

    // 使用 curl 获取 HTML 内容
    const curlCommand = `curl -s -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" "${githubUrl}"`;
    const htmlText = execSync(curlCommand, { encoding: 'utf-8' });

    if (!htmlText || htmlText.length === 0) {
      throw new Error('Empty response from GitHub');
    }

    // 解析 GitHub 贡献图数据
    // GitHub 使用 SVG 格式，每个 <rect> 元素包含：
    // - data-date: 日期 (YYYY-MM-DD)
    // - data-count: 精确贡献数（如果存在）
    // - data-level: 贡献等级（0-4）
    // 注意：GitHub 的 SVG 结构可能包含自闭合或带子元素的 rect

    const contributionsMap = new Map();
    const rectPattern = /<rect[^>]*data-date="([^"]+)"[^>]*>/g;
    let match;

    // 统一处理所有 rect 元素，无论是自闭合还是带子元素
    while ((match = rectPattern.exec(htmlText)) !== null) {
      const rectStart = match[0];
      const date = match[1];

      // 跳过已处理的日期
      if (contributionsMap.has(date)) continue;

      let count = 0;

      // 优先级1: data-count（最精确）
      const countMatch = rectStart.match(/data-count="(\d+)"/);
      if (countMatch) {
        count = parseInt(countMatch[1], 10) || 0;
      } else {
        // 优先级2: 从完整的 rect 元素中提取 title 内容
        // 找到完整的 rect 元素（包括可能的子元素和结束标签）
        const rectEndIndex = htmlText.indexOf('>', rectPattern.lastIndex - 1);
        if (rectEndIndex !== -1) {
          const isSelfClosing = rectStart.trim().endsWith('/>');
          let fullRect = '';

          if (isSelfClosing) {
            fullRect = rectStart;
          } else {
            // 查找结束标签
            const endIndex = htmlText.indexOf('</rect>', rectEndIndex);
            if (endIndex !== -1) {
              fullRect = htmlText.substring(match.index, endIndex + 8); // +8 for </rect>
            } else {
              fullRect = rectStart;
            }
          }

          // 从完整 rect 中提取 title
          const titleMatch = fullRect.match(/<title>([^<]*)<\/title>/);
          if (titleMatch) {
            const titleText = titleMatch[1];
            const contributionMatch = titleText.match(/(\d+)\s+contribution/i);
            if (contributionMatch) {
              count = parseInt(contributionMatch[1], 10) || 0;
            } else if (titleText.toLowerCase().includes('no contribution')) {
              count = 0;
            }
          }

          // 如果还没找到，尝试 title 属性
          if (count === 0) {
            const titleAttrMatch = fullRect.match(/title="([^"]*)"/);
            if (titleAttrMatch) {
              const titleStr = titleAttrMatch[1];
              const contributionMatch = titleStr.match(/(\d+)\s+contribution/i);
              if (contributionMatch) {
                count = parseInt(contributionMatch[1], 10) || 0;
              } else if (titleStr.toLowerCase().includes('no contribution')) {
                count = 0;
              }
            }
          }
        }

        // 优先级3: 使用 data-level（最后手段）
        if (count === 0) {
          const levelMatch = rectStart.match(/data-level="(\d+)"/);
          if (levelMatch) {
            const level = parseInt(levelMatch[1], 10);
            // 使用 GitHub 官方的颜色映射对应的贡献范围
            if (level === 1) count = 1;
            else if (level === 2) count = 4; // 2-6 范围的中间值
            else if (level === 3) count = 8; // 7-11 范围的中间值
            else if (level === 4) count = 12; // 12+ 范围的起始值
          }
        }
      }

      contributionsMap.set(date, count);
    }

    // 转换为数组并按日期排序
    const contributions = Array.from(contributionsMap.entries())
      .map(([date, count]) => ({ date, count }))
      .sort((a, b) => a.date.localeCompare(b.date));

    // 如果没有找到数据，尝试旧的 table 格式作为备选
    if (contributions.length === 0) {
      const tdPattern = /<td[^>]*data-date="([^"]+)"[^>]*data-level="(\d+)"[^>]*>/g;
      while ((match = tdPattern.exec(htmlText)) !== null) {
        const date = match[1];
        const level = parseInt(match[2], 10);

        if (contributionsMap.has(date)) continue;

        let count = 0;
        if (level === 1) count = 1;
        else if (level === 2) count = 4;
        else if (level === 3) count = 8;
        else if (level === 4) count = 12;

        contributionsMap.set(date, count);
      }

      // 重新生成排序后的数组
      const fallbackContributions = Array.from(contributionsMap.entries())
        .map(([date, count]) => ({ date, count }))
        .sort((a, b) => a.date.localeCompare(b.date));

      if (fallbackContributions.length > 0) {
        contributions.length = 0;
        contributions.push(...fallbackContributions);
      }
    }

    if (contributions.length === 0) {
      console.error('[Build] No contributions found in HTML');
      return { contributions: [], totalContributions: 0 };
    }

    const totalContributions = contributions.reduce((sum, c) => sum + c.count, 0);
    console.log(`[Build] Parsed ${contributions.length} contribution days, total: ${totalContributions}`);

    return { contributions, totalContributions };

  } catch (error) {
    console.error('[Build] Error fetching GitHub contributions:', error);
    return { contributions: [], totalContributions: 0, error: error.message };
  }
}

async function main() {
  try {
    const { contributions, totalContributions, error } = await fetchGitHubContributions();

    // 保存到 public 目录
    const outputPath = join(__dirname, '..', 'public', 'api', 'github-contributions.json');
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