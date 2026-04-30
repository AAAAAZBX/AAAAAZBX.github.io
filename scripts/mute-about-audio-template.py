"""One-off friendly: mute T-Rex base64 audio in about.astro (run via npm script if needed)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "pages" / "about.astro"
s = path.read_text(encoding="utf-8")
s2 = re.sub(
    r'<template id="audio-resources">.*?</template>',
    '<template id="audio-resources" data-muted="true"></template>',
    s,
    flags=re.DOTALL,
)
path.write_text(s2, encoding="utf-8")
print("OK:", path)
