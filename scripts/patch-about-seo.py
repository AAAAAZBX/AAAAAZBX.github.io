"""UTF-8-safe patches for about.astro (BASE_URL, Layout description, contributions JSON path)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "pages" / "about.astro"
s = path.read_text(encoding="utf-8")

block_old = 'import Layout from "../layouts/Layout.astro";\n\nconst profile'
block_new = """import Layout from "../layouts/Layout.astro";
import { withBase } from "../lib/base-url";

const base = import.meta.env.BASE_URL;
const contributionsDataUrl = withBase(base, "data/github-contributions.json");
const logoUrl = withBase(base, "logo.png");

const profile"""
if block_old in s and "withBase" not in s:
    s = s.replace(block_old, block_new, 1)

layout_old = "<Layout title={`About — ${profile.name}`}>"
layout_new = """<Layout
  title={`About — ${profile.name}`}
  description={`${profile.name} — ${profile.title}. Nanjing, China.`}
>"""
if layout_old in s:
    s = s.replace(layout_old, layout_new, 1)

s = s.replace(
    '<img src="/logo.png" alt={profile.name} class="avatar" />',
    '<img src={logoUrl} alt={profile.name} class="avatar" />',
    1,
)
s = s.replace(
    '<link rel="stylesheet" href="/styles/index.css" />',
    '<link rel="stylesheet" href={withBase(base, "styles/index.css")} />',
    1,
)
s = s.replace(
    '<link rel="stylesheet" href="/t-rex-runner-gh-pages/index.css" />',
    '<link rel="stylesheet" href={withBase(base, "t-rex-runner-gh-pages/index.css")} />',
    1,
)
s = s.replace(
    'src="/t-rex-runner-gh-pages/assets/default_100_percent/100-offline-sprite.png"',
    'src={withBase(base, "t-rex-runner-gh-pages/assets/default_100_percent/100-offline-sprite.png")}',
    1,
)
s = s.replace(
    'src="/t-rex-runner-gh-pages/assets/default_200_percent/200-offline-sprite.png"',
    'src={withBase(base, "t-rex-runner-gh-pages/assets/default_200_percent/200-offline-sprite.png")}',
    1,
)
s = s.replace(
    '<script is:inline src="/t-rex-runner-gh-pages/index.js"></script>',
    '<script is:inline src={withBase(base, "t-rex-runner-gh-pages/index.js")}></script>',
    1,
)

old_script = "<script is:inline>\n  (function() {"
if old_script in s and "define:vars" not in s:
    s = s.replace(
        old_script,
        '<script is:inline define:vars={{ contributionsDataUrl }}>\n  (function() {',
        1,
    )

s = s.replace(
    "const apiUrl = '/api/github-contributions.json';",
    "const apiUrl = contributionsDataUrl;",
    1,
)

esc_snippet = """    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = String(text);
      return div.innerHTML;
    }

"""
if esc_snippet.strip() not in s:
    s = s.replace(
        "    async function loadContributions() {",
        esc_snippet + "    async function loadContributions() {",
        1,
    )

s = s.replace(
    "graphEl.innerHTML = `<p style=\"color: #666; padding: 20px;\">Error: ${error.message}. <a href=\"https://github.com/AAAAAZBX\" target=\"_blank\">View on GitHub</a></p>`;",
    "graphEl.innerHTML = '<p style=\"color: #666; padding: 20px;\">Error: ' + escapeHtml(error.message) + '. <a href=\"https://github.com/AAAAAZBX\" target=\"_blank\">View on GitHub</a></p>';",
    1,
)

path.write_text(s, encoding="utf-8")
print("patched", path)
