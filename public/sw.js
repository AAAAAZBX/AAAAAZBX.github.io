/**
 * 轻量 Service Worker：预缓存全站常用顶图，并对同站图片请求使用 cache-first，
 * 减轻多页跳转时背景图重复下载（需 HTTPS 或 localhost）。
 */
const CACHE_NAME = 'tremendous-assets-v1';
const swUrl = self.location.href;

function resolveAsset(name) {
  return new URL(name, swUrl).href;
}

const PRECACHE_URLS = [resolveAsset('site-profile.png'), resolveAsset('logo.png')];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      Promise.all(
        PRECACHE_URLS.map((url) =>
          fetch(url, { mode: 'cors', credentials: 'same-origin' })
            .then((res) => {
              if (res.ok) return cache.put(url, res);
            })
            .catch(() => {})
        )
      )
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME && key.startsWith('tremendous-assets-')) {
            return caches.delete(key);
          }
          return Promise.resolve();
        })
      ).then(() => self.clients.claim())
    )
  );
});

function isSameOriginImageRequest(request, url) {
  if (request.method !== 'GET') return false;
  if (url.origin !== self.location.origin) return false;
  if (request.destination === 'image') return true;
  return /\.(png|jpe?g|gif|webp|avif|svg)$/i.test(url.pathname);
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  let url;
  try {
    url = new URL(req.url);
  } catch {
    return;
  }
  if (!isSameOriginImageRequest(req, url)) return;

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      const hit = await cache.match(req, { ignoreSearch: false });
      if (hit) return hit;
      try {
        const res = await fetch(req);
        if (res.ok && res.type === 'basic') {
          try {
            cache.put(req, res.clone());
          } catch (_) {}
        }
        return res;
      } catch (e) {
        return cache.match(req);
      }
    })
  );
});
