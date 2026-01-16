// 构建时获取最近提交的脚本
import { execSync } from 'child_process';
import { writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

try {
  console.log('[Build] Fetching recent commits...');
  
  // 获取最近 10 条提交
  // 格式: hash|date|message
  const gitLogCommand = 'git log --pretty=format:"%H|%ai|%s" -10';
  
  let commits = [];
  try {
    const output = execSync(gitLogCommand, { 
      encoding: 'utf-8',
      cwd: join(__dirname, '..')
    });
    
    const lines = output.trim().split('\n').filter(line => line.trim());
    
    commits = lines.map(line => {
      const [hash, date, ...messageParts] = line.split('|');
      const message = messageParts.join('|'); // 处理消息中可能包含 | 的情况
      
      return {
        hash: hash.trim(),
        date: date.trim(),
        message: message.trim(),
      };
    });
    
    console.log(`[Build] Found ${commits.length} recent commits`);
  } catch (error) {
    console.error('[Build] Error running git log:', error.message);
    // 如果 git log 失败，返回空数组
    commits = [];
  }
  
  // 保存到 public 目录
  const outputPath = join(__dirname, '..', 'public', 'api', 'recent-commits.json');
  const outputDir = dirname(outputPath);
  mkdirSync(outputDir, { recursive: true });
  
  const data = {
    commits,
    timestamp: new Date().toISOString(),
  };
  
  writeFileSync(outputPath, JSON.stringify(data, null, 2), 'utf-8');
  console.log(`[Build] Saved recent commits to ${outputPath}`);
  
} catch (error) {
  console.error('[Build] Error:', error);
  process.exit(1);
}
