// Versioned covers are immutable; other images refresh in the background.
const CACHE_NAME = 'tremendous-assets-v3';
const MAX_IMAGES = 80;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((key) => key.startsWith('tremendous-assets-') && key !== CACHE_NAME)
      .map((key) => caches.delete(key)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin || request.destination !== 'image') return;
  const cachePromise = caches.open(CACHE_NAME);
  const cachedPromise = cachePromise.then((cache) => cache.match(request));
  const immutable = /site-profile-\d+-[a-f0-9]{12}\.webp$/.test(url.pathname);
  const refresh = (async () => {
    const cached = await cachedPromise;
    if (immutable && cached) return cached;
    const response = await fetch(request);
    if (response.ok && response.type === 'basic') {
      const cache = await cachePromise;
      await cache.put(request, response.clone());
      const keys = await cache.keys();
      await Promise.all(keys.slice(0, Math.max(0, keys.length - MAX_IMAGES)).map((key) => cache.delete(key)));
    }
    return response;
  })();
  event.waitUntil(refresh.catch(() => {}));
  event.respondWith((async () => (await cachedPromise) || refresh)());
});
