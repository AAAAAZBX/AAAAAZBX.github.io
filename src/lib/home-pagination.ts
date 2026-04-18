/** 首页文章列表每页条数（对齐 Fluid / 林正博客式分页） */
export const HOME_POSTS_PER_PAGE = 10;

export function getHomeTotalPages(postCount: number): number {
  return Math.max(1, Math.ceil(postCount / HOME_POSTS_PER_PAGE));
}

export function sliceHomePostsPage<T>(items: T[], page: number): T[] {
  const p = Math.max(1, Math.floor(page));
  const start = (p - 1) * HOME_POSTS_PER_PAGE;
  return items.slice(start, start + HOME_POSTS_PER_PAGE);
}
