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
    // GitHub 2024+ 新结构：
    // - <td> 元素包含 data-date 和 id
    // - <tool-tip> 元素包含精确贡献数，通过 for 属性关联到 td 的 id

    const contributionsMap = new Map();
    let match;

    // 步骤1: 解析所有 <td> 元素，建立 id -> date 的映射
    const idToDateMap = new Map();
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

    console.log('[Build] Found', idToDateMap.size, 'td elements with data-date');

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
        
        contributionsMap.set(date, count);
      }
    }

    // 转换为数组并按日期排序
    const contributions = Array.from(contributionsMap.entries())
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