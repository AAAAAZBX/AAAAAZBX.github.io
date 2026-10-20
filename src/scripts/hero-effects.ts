// Shared enhancement: full titles remain available when scripts are unavailable.
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const finishTyping: Array<() => void> = [];

document.querySelectorAll<HTMLElement>('[data-typed-text]').forEach((element) => {
  const text = element.dataset.typedText;
  if (!text || reducedMotion.matches) return;
  const characters = Array.from(text);
  const cursor = document.createElement('span');
  cursor.className = 'typed-cursor';
  cursor.textContent = '_';
  cursor.setAttribute('aria-hidden', 'true');
  element.after(cursor);
  element.setAttribute('aria-hidden', 'true');
  element.parentElement?.setAttribute('aria-label', text);
  let index = 0;
  element.textContent = '';
  const timer = window.setInterval(() => {
    element.textContent = characters.slice(0, ++index).join('');
    if (index >= characters.length) window.clearInterval(timer);
  }, 50);
  finishTyping.push(() => {
    window.clearInterval(timer);
    element.textContent = text;
    cursor.remove();
  });
});

const hero = document.querySelector<HTMLElement>('.fluid-index-header-inner');
const background = hero?.querySelector<HTMLElement>('.fluid-banner-photo-mode .fluid-banner-bg');
const homeBoard = document.querySelector<HTMLElement>('#board.fluid-board-home');
const compactBoard = hero?.classList.contains('fluid-header-inner-compact')
  ? document.querySelector<HTMLElement>('main #board.fluid-board-surface') : null;
const board = homeBoard ?? compactBoard;

if (hero && (background || board)) {
  let ticking = false;
  let lastScroll = -1;
  const update = () => {
    ticking = false;
    if (reducedMotion.matches) {
      background?.style.removeProperty('transform');
      board?.style.removeProperty('transform');
      return;
    }
    const top = Math.max(0, window.scrollY);
    const height = hero.clientHeight || 1;
    // Transforms are capped after the cover; avoid work while reading the article list.
    const scroll = Math.min(top, height);
    if (scroll === lastScroll) return;
    lastScroll = scroll;
    if (background) {
      background.style.transform = `translate3d(0,${Math.round(scroll * 0.28)}px,0) scale(${(1 + scroll / height * 0.04).toFixed(3)})`;
    }
    if (board) {
      const progress = Math.min(scroll / Math.max(height * (homeBoard ? 0.52 : 0.85), homeBoard ? 72 : 120), 1);
      const blend = homeBoard ? progress : progress * progress * (3 - 2 * progress);
      board.style.transform = `translate3d(0,${-Math.round(blend * (homeBoard ? 112 : 88))}px,0)`;
    }
  };
  const requestUpdate = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(update);
  };
  window.addEventListener('scroll', requestUpdate, { passive: true });
  window.addEventListener('resize', () => { lastScroll = -1; requestUpdate(); });
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) finishTyping.forEach((finish) => finish());
    lastScroll = -1;
    requestUpdate();
  });
  update();
}
