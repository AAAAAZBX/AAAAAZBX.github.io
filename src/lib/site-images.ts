import images from '../generated/site-images.json';
import { withBase } from './base-url';

export function defaultHeroImage(base: string): string {
  return withBase(base, images['1920']);
}

export function heroImageSrcSet(base: string): string {
  return `${withBase(base, images['960'])} 960w, ${defaultHeroImage(base)} 1920w`;
}
