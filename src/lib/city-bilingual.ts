/**
 * 将 ipapi 等返回的 city / region 拆成**始终可读的两栏**：中文城市名、英文城市名。
 * 在仅有单语时用语种检测 + 内置常见城市表补全；仍无法确认时用「—」。
 */

const RE_CJK = /[\u4e00-\u9fff\u3400-\u4dbf]/;

/** 英文小写主键（去标点、归一化空格）→ 中文惯用名 */
export const EN_TO_ZH: Record<string, string> = {
  // 中国主要城市
  beijing: "北京",
  shanghai: "上海",
  guangzhou: "广州",
  shenzhen: "深圳",
  chengdu: "成都",
  hangzhou: "杭州",
  wuhan: "武汉",
  xian: "西安",
  xianyang: "咸阳",
  nanjing: "南京",
  tianjin: "天津",
  chongqing: "重庆",
  suzhou: "苏州",
  zhengzhou: "郑州",
  changsha: "长沙",
  dongguan: "东莞",
  qingdao: "青岛",
  shenyang: "沈阳",
  ningbo: "宁波",
  kunming: "昆明",
  wuxi: "无锡",
  foshan: "佛山",
  hefei: "合肥",
  dalian: "大连",
  jinan: "济南",
  changchun: "长春",
  shijiazhuang: "石家庄",
  harbin: "哈尔滨",
  fuzhou: "福州",
  nanchang: "南昌",
  xiamen: "厦门",
  nanning: "南宁",
  taiyuan: "太原",
  yantai: "烟台",
  zhuhai: "珠海",
  nantong: "南通",
  weifang: "潍坊",
  jinhua: "金华",
  xuzhou: "徐州",
  changzhou: "常州",
  wenzhou: "温州",
  quanzhou: "泉州",
  wuhu: "芜湖",
  hohhot: "呼和浩特",
  urumqi: "乌鲁木齐",
  lhasa: "拉萨",
  yinchuan: "银川",
  xining: "西宁",
  haikou: "海口",
  sanya: "三亚",
  lijiang: "丽江",
  guilin: "桂林",
  mianyang: "绵阳",
  zibo: "淄博",
  tangshan: "唐山",
  baoding: "保定",
  lanzhou: "兰州",
  luoyang: "洛阳",
  yancheng: "盐城",
  huzhou: "湖州",
  jiaxing: "嘉兴",
  taizhou: "泰州",
  "hong kong": "香港",
  macau: "澳门",
  macao: "澳门",
  kaohsiung: "高雄",
  taipei: "台北",
  taichung: "台中",
  tainan: "台南",
  // 美国与其它常见访问来源
  "new york": "纽约",
  "new york city": "纽约",
  "los angeles": "洛杉矶",
  chicago: "芝加哥",
  houston: "休斯敦",
  phoenix: "菲尼克斯",
  philadelphia: "费城",
  "san antonio": "圣安东尼奥",
  "san diego": "圣迭戈",
  dallas: "达拉斯",
  "san jose": "圣何塞",
  austin: "奥斯汀",
  "san francisco": "旧金山",
  seattle: "西雅图",
  denver: "丹佛",
  boston: "波士顿",
  atlanta: "亚特兰大",
  miami: "迈阿密",
  detroit: "底特律",
  minneapolis: "明尼阿波利斯",
  tampa: "坦帕",
  orlando: "奥兰多",
  cleveland: "克利夫兰",
  pittsburgh: "匹兹堡",
  stlouis: "圣路易斯",
  portland: "波特兰",
  lasvegas: "拉斯维加斯",
  nashville: "纳什维尔",
  columbus: "哥伦布",
  "washington, dc": "华盛顿",
  washington: "华盛顿",
  vancouver: "温哥华",
  toronto: "多伦多",
  montreal: "蒙特利尔",
  calgary: "卡尔加里",
  ottawa: "渥太华",
  london: "伦敦",
  manchester: "曼彻斯特",
  birmingham: "伯明翰",
  edinburgh: "爱丁堡",
  dublin: "都柏林",
  paris: "巴黎",
  lyon: "里昂",
  marseille: "马赛",
  berlin: "柏林",
  munich: "慕尼黑",
  hamburg: "汉堡",
  frankfurt: "法兰克福",
  amsterdam: "阿姆斯特丹",
  brussels: "布鲁塞尔",
  vienna: "维也纳",
  zurich: "苏黎世",
  geneva: "日内瓦",
  rome: "罗马",
  milan: "米兰",
  madrid: "马德里",
  barcelona: "巴塞罗那",
  lisbon: "里斯本",
  stockholm: "斯德哥尔摩",
  copenhagen: "哥本哈根",
  oslo: "奥斯陆",
  helsinki: "赫尔辛基",
  warsaw: "华沙",
  prague: "布拉格",
  budapest: "布达佩斯",
  athens: "雅典",
  moscow: "莫斯科",
  "st petersburg": "圣彼得堡",
  "saint petersburg": "圣彼得堡",
  kiev: "基辅",
  dubai: "迪拜",
  "abu dhabi": "阿布扎比",
  singapore: "新加坡",
  tokyo: "东京",
  yokohama: "横滨",
  osaka: "大阪",
  nagoya: "名古屋",
  kyoto: "京都",
  fukuoka: "福冈",
  sapporo: "札幌",
  seoul: "首尔",
  busan: "釜山",
  sydney: "悉尼",
  melbourne: "墨尔本",
  brisbane: "布里斯班",
  perth: "珀斯",
  auckland: "奥克兰",
  wellington: "惠灵顿",
  bangkok: "曼谷",
  jakarta: "雅加达",
  hanoi: "河内",
  "ho chi minh": "胡志明市",
  manila: "马尼拉",
  mumbai: "孟买",
  delhi: "新德里",
  bengaluru: "班加罗尔",
  "kuala lumpur": "吉隆坡",
  "saint louis": "圣路易斯",
};

function normalizeEnKey(s: string): string {
  return s
    .trim()
    .toLowerCase()
    .replace(/[.']/g, "")
    .replace(/\s+/g, " ");
}

function titleCaseEnKey(key: string): string {
  return key
    .split(/[\s-]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

/** 由 EN_TO_ZH 反查：中文（含常见「市/省」后缀）→ 英文 key */
function buildZhToEn(): Record<string, string> {
  const m: Record<string, string> = {};
  for (const [en, zh] of Object.entries(EN_TO_ZH)) {
    if (!zh) continue;
    if (!m[zh]) m[zh] = en;
  }
  const sup: Record<string, string> = {
    北京市: "beijing",
    上海市: "shanghai",
    广州市: "guangzhou",
    深圳市: "shenzhen",
    香港特别行政区: "hong kong",
    澳门特别行政区: "macau",
  };
  for (const [z, e] of Object.entries(sup)) {
    m[z] = e;
  }
  return m;
}

const ZH_TO_EN = buildZhToEn();

const EM = "—";

function lookupZhFromEn(en: string): string {
  const k0 = normalizeEnKey(en);
  if (EN_TO_ZH[k0]) {
    return EN_TO_ZH[k0];
  }
  const first = k0.split(" ")[0] ?? "";
  if (first && EN_TO_ZH[first]) {
    return EN_TO_ZH[first];
  }
  return "";
}

function lookupEnFromZh(zh: string): string {
  if (ZH_TO_EN[zh]) {
    return titleCaseEnKey(ZH_TO_EN[zh]);
  }
  const stripped = zh.replace(/(特别行政区|市|省|区|县|州)$/, "");
  if (stripped !== zh && ZH_TO_EN[stripped]) {
    return titleCaseEnKey(ZH_TO_EN[stripped]);
  }
  return "";
}

/**
 * 返回中文名、英文城市名展示串；若无法补全，对应为「—」。
 * 不用于省/州（region 在纯英文时可能是州名，不强行当地市译）。
 */
export function resolveCityBilingual(
  city: string | null,
  region: string | null
): { zh: string; en: string } {
  const c = (city ?? "").trim();
  const reg = (region ?? "").trim();

  const takeZh = (s: string) => (s && RE_CJK.test(s) ? s : "");

  let zh = takeZh(c) || takeZh(reg) || "";
  let en = "";

  if (c && !RE_CJK.test(c) && /[A-Za-z]/.test(c)) {
    en = c;
  } else if (!en && reg && !RE_CJK.test(reg) && /[A-Za-z]/.test(reg)) {
    if (!c) {
      en = reg;
    }
  }

  if (!zh && !en) {
    if (c) {
      if (RE_CJK.test(c)) {
        zh = c;
      } else {
        en = c;
      }
    } else if (reg) {
      if (RE_CJK.test(reg)) {
        zh = reg;
      } else {
        en = reg;
      }
    }
  }

  if (en && !zh) {
    zh = lookupZhFromEn(en) || EM;
  }
  if (zh && zh !== EM && !en) {
    en = lookupEnFromZh(zh) || EM;
  }
  if (en && en !== EM && (!zh || zh === EM)) {
    const z = lookupZhFromEn(en);
    if (z) {
      zh = z;
    }
  }

  if (!zh) {
    zh = EM;
  }
  if (!en) {
    en = EM;
  }
  return { zh, en };
}

/**
 * 单行展示为「英文, 中文」（如 `Beijing, 北京`）；仅有一种则只显示该种；均无时为「—」。
 */
export function formatCityPair(zh: string, en: string): string {
  const z = zh === EM || !String(zh).trim() ? "" : String(zh).trim();
  const e = en === EM || !String(en).trim() ? "" : String(en).trim();
  if (!z && !e) {
    return EM;
  }
  if (z && e) {
    if (z === e) {
      return z;
    }
    return `${e}, ${z}`;
  }
  return z || e;
}
