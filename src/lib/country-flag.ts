/**
 * ISO 3166-1 alpha-2 → 国旗展示（矢量 SVG，flag-icons / jsDelivr）。
 * 小位图在高分屏上易糊；SVG 任意缩放仍清晰。Windows 上国旗 emoji 常退化成字母，故不用 emoji。
 * 国家英文名与 ipapi.co / 常见 GeoIP 的 country_name 对齐，用于无 country_code 时的回退。
 */

/** lipis/flag-icons，4:3 资源；与 ISO alpha-2 文件名一致（如 gb.svg） */
const FLAG_ICONS_4X3 =
  "https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/flags/4x3";

const NAME_TO_ISO: Record<string, string> = {
  china: "CN",
  中国: "CN",
  "people's republic of china": "CN",
  "united states": "US",
  "united states of america": "US",
  美国: "US",
  usa: "US",
  japan: "JP",
  日本: "JP",
  "south korea": "KR",
  korea: "KR",
  "korea, republic of": "KR",
  germany: "DE",
  france: "FR",
  "united kingdom": "GB",
  uk: "GB",
  "great britain": "GB",
  england: "GB",
  canada: "CA",
  australia: "AU",
  india: "IN",
  brazil: "BR",
  russia: "RU",
  "russian federation": "RU",
  italy: "IT",
  spain: "ES",
  mexico: "MX",
  indonesia: "ID",
  netherlands: "NL",
  turkey: "TR",
  "türkiye": "TR",
  saudiarabia: "SA",
  "saudi arabia": "SA",
  switzerland: "CH",
  poland: "PL",
  belgium: "BE",
  argentina: "AR",
  sweden: "SE",
  norway: "NO",
  austria: "AT",
  thailand: "TH",
  unitedarabemirates: "AE",
  "united arab emirates": "AE",
  vietnam: "VN",
  "viet nam": "VN",
  iran: "IR",
  "iran, islamic republic of": "IR",
  israel: "IL",
  singapore: "SG",
  malaysia: "MY",
  philippines: "PH",
  egypt: "EG",
  pakistan: "PK",
  bangladesh: "BD",
  nigeria: "NG",
  southafrica: "ZA",
  "south africa": "ZA",
  ukraine: "UA",
  colombia: "CO",
  chile: "CL",
  finland: "FI",
  romania: "RO",
  czechia: "CZ",
  "czech republic": "CZ",
  portugal: "PT",
  greece: "GR",
  hungary: "HU",
  newzealand: "NZ",
  "new zealand": "NZ",
  ireland: "IE",
  denmark: "DK",
  croatia: "HR",
  serbia: "RS",
  bulgaria: "BG",
  slovakia: "SK",
  slovenia: "SI",
  lithuania: "LT",
  latvia: "LV",
  estonia: "EE",
  luxembourg: "LU",
  iceland: "IS",
  qatar: "QA",
  kuwait: "KW",
  kazakhstan: "KZ",
  uzbekistan: "UZ",
  peru: "PE",
  venezuela: "VE",
  ecuador: "EC",
  morocco: "MA",
  kenya: "KE",
  ethiopia: "ET",
  ghana: "GH",
  "hong kong": "HK",
  香港: "HK",
  macao: "MO",
  macau: "MO",
  taiwan: "TW",
  台湾: "TW",
  "taiwan, province of china": "TW",
};

function normalizeCountryKey(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replaceAll(".", "")
    .replace(/\s+/g, " ");
}

/** 合法两位大写 ISO，否则 null */
export function normalizeAlpha2(code: string | null | undefined): string | null {
  const c = String(code ?? "")
    .trim()
    .toUpperCase();
  if (c.length !== 2 || !/^[A-Z]{2}$/.test(c)) return null;
  return c;
}

/** 矢量 SVG；intrinsic 4:3，具体显示尺寸由 CSS 控制 */
export function flagImgHtml(iso: string | null | undefined): string {
  const c = normalizeAlpha2(iso);
  if (!c) return "";
  const cl = c.toLowerCase();
  const src = `${FLAG_ICONS_4X3}/${cl}.svg`;
  return `<img class="visitor-insights__flag-img" src="${src}" width="32" height="24" alt="" loading="lazy" decoding="async" referrerpolicy="no-referrer" onerror="this.style.display='none'" />`;
}

export function inferAlpha2FromCountryName(countryName: string | null | undefined): string | null {
  if (!countryName?.trim()) return null;
  const k = normalizeCountryKey(countryName);
  return NAME_TO_ISO[k] ?? null;
}

/** 优先用存储的 country_code，否则从英文国名推断 */
export function resolveCountryAlpha2(
  countryName: string | null | undefined,
  countryCode: string | null | undefined,
): string | null {
  const fromCode = normalizeAlpha2(countryCode);
  if (fromCode) return fromCode;
  return inferAlpha2FromCountryName(countryName);
}
