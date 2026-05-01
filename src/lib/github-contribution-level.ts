/**
 * GitHub 个人页贡献热力图的颜色档位（与「某天有几个 contribution」相关，且是相对比例，不是固定阈值）。
 *
 * 官方 GraphQL 文档里写成 ContributionLevel / quartile（
 * https://docs.github.com/en/graphql/reference/enums#contributionlevel ），
 * 但与「把日历里的天数均分成四组」并不一致；实测更接近下面规则（与 GitHub 渲染对照）：
 * https://stackoverflow.com/questions/75431524
 *
 * - 令 `maxDaily` = 当前展示窗口内「单日贡献数」的最大值（含历史上有活动的日期）。
 * - `q = maxDaily / 4`（把 `(0, maxDaily]` 均分为四段）。
 * - 0 贡献：NONE（灰色）。
 * - 非零：`>= 3q`、`>= 2q`、`>= q`、否则最浅一档（共 4 档绿色）。
 *
 * 因此当你某天突然有很多提交、`maxDaily` 变大时，**更早的同贡献数日期可能掉到更低的档位**，颜色会变浅——这是正常现象，不是缓存错乱。
 */

export type GithubContributionLevel = 0 | 1 | 2 | 3 | 4;

/** 与 GitHub 默认亮色主题下的方块填充色一致（生产环境近似值）。 */
export const GITHUB_CONTRIBUTION_FILL_LIGHT: readonly string[] = [
  "#ebedf0",
  "#9be9a8",
  "#40c463",
  "#30a14e",
  "#216e39",
] as const;

export function githubContributionLevel(count: number, maxDaily: number): GithubContributionLevel {
  const n = Math.trunc(Number(count));
  if (!Number.isFinite(n) || n <= 0) return 0;
  const max = Math.max(0, Math.trunc(Number(maxDaily)));
  if (max <= 0) return 1;
  const q = max / 4;
  if (n >= 3 * q) return 4;
  if (n >= 2 * q) return 3;
  if (n >= q) return 2;
  return 1;
}

export function githubContributionFillColor(level: GithubContributionLevel): string {
  return GITHUB_CONTRIBUTION_FILL_LIGHT[level] ?? GITHUB_CONTRIBUTION_FILL_LIGHT[0];
}
