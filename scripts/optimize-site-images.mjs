import sharp from 'sharp';
import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const source = await readFile(new URL('public/site-profile.png', root));
const version = createHash('sha256').update(source).update('webp-v1-quality82').digest('hex').slice(0, 12);
await mkdir(new URL('public/images/', root), { recursive: true });
const images = {};
for (const width of [960, 1920]) {
  const name = `site-profile-${width}-${version}.webp`;
  await writeFile(new URL(`public/images/${name}`, root), await sharp(source)
    .rotate().resize({ width, withoutEnlargement: true }).webp({ quality: 82 }).toBuffer());
  images[width] = `images/${name}`;
}
await mkdir(new URL('src/generated/', root), { recursive: true });
await writeFile(new URL('src/generated/site-images.json', root), `${JSON.stringify(images, null, 2)}\n`);
console.log('Generated responsive WebP covers.');
