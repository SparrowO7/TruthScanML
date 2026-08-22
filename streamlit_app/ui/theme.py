"""Single source of visual styling for every page — Apple Liquid Glass.

Reference: user's "liquid glass form" demo (3-layer recipe):
  bend layer  → backdrop-filter blur + filter:url() displacement (refraction)
  face layer  → soft drop shadows for depth
  edge layer  → white inset glossy edge highlights

In Chromium, ``url()`` is NOT valid inside ``backdrop-filter`` — the
displacement must be applied as an element-level ``filter`` on a separate
pseudo-layer. That layer uses ``z-index:-1`` so it sits between the card
background and the content: the page background behind the card gets warped,
but the card text (painted later) stays crisp.

The background carries several large, slowly drifting color blobs so the
glass always has something visible to refract.
"""

from textwrap import dedent

import streamlit as st


# SVG displacement filter for glass refraction. Declared once per page.
_GLASS_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0"
     style="position:absolute;pointer-events:none" aria-hidden="true">
  <filter id="tsGlass" x="0" y="0" width="100%" height="100%"
          filterUnits="objectBoundingBox"
          color-interpolation-filters="sRGB">
    <feTurbulence type="fractalNoise" baseFrequency="0.004 0.006"
                  numOctaves="2" seed="92" result="turbulence"/>
    <feDisplacementMap in="SourceGraphic" in2="turbulence" scale="48"
                       xChannelSelector="R" yChannelSelector="G"/>
  </filter>
</svg>
"""

# Ambient background blobs injected as a fixed layer behind the content.
_AMBIENT_HTML = """
<div class="ts-ambient" aria-hidden="true">
  <div class="ts-blob ts-blob-a"></div>
  <div class="ts-blob ts-blob-b"></div>
  <div class="ts-blob ts-blob-c"></div>
</div>
"""


_THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

/* ==========================================================================
   Tokens — derived from Streamlit's theme variables; semantic hues fixed.
   ========================================================================== */

:root {
    --ts-bg:          var(--background, #0c101c);
    --ts-text:        var(--text-color, #edf0fa);
    --ts-accent:      var(--primary, #7c8cf8);
    --ts-text-muted:  color-mix(in srgb, var(--text-color, #ffffff) 60%, transparent);
    --ts-tint:        rgba(255, 255, 255, 0.045);  /* card face tint */
    --ts-tint-strong: rgba(255, 255, 255, 0.08);
    --ts-edge-a:      rgba(255, 255, 255, 0.34);   /* glossy edge, top-left */
    --ts-edge-b:      rgba(255, 255, 255, 0.14);   /* glossy edge, bottom-right */
    --ts-ring-track:  color-mix(in srgb, var(--text-color, #ffffff) 12%, transparent);
    --ts-ring-inner:  var(--background, #12172b);
    --ts-shadow:      0 10px 26px rgba(8, 10, 24, 0.16), 0 26px 64px rgba(8, 10, 24, 0.20);
    --ts-shadow-hover: 0 12px 30px rgba(8, 10, 24, 0.20), 0 34px 76px rgba(8, 10, 24, 0.26);
    --ts-pos:         #34d58b;
    --ts-neg:         #ff6b6b;
    --ts-amb:         #ffb224;
    --ts-radius:      24px;
}

/* ---------- Base ---------- */

html, body, .stApp, [data-testid="stAppViewContainer"] {
    color: var(--ts-text);
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, [data-testid="stMetricLabel"] {
    font-family: 'Sora', 'Inter', sans-serif;
}

h1 { font-weight: 700; letter-spacing: -0.015em; color: var(--ts-text); }

[data-testid="stHeader"] { background: transparent; }

[data-testid="stCaptionContainer"], .stCaption, p.caption {
    color: var(--ts-text-muted) !important;
}

/* ==========================================================================
   AMBIENT BACKGROUND — big, colorful, slowly moving. Glass needs something
   to refract; a flat background hides the whole effect.
   ========================================================================== */

.stApp, [data-testid="stAppViewContainer"] {
    background: var(--ts-bg);
}

.ts-ambient {
    position: fixed;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}

.ts-blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.26;
    will-change: transform;
}

.ts-blob-a {
    width: 48vw; height: 48vw;
    left: -8vw; top: -10vh;
    background: radial-gradient(circle at 30% 30%, #5c6bc0, transparent 70%);
    animation: blobA 36s ease-in-out infinite alternate;
}

.ts-blob-b {
    width: 42vw; height: 42vw;
    right: -6vw; top: 16vh;
    background: radial-gradient(circle at 60% 40%, #44788f, transparent 70%);
    animation: blobB 44s ease-in-out infinite alternate;
    opacity: 0.22;
}

.ts-blob-c {
    width: 40vw; height: 40vw;
    left: 26vw; bottom: -16vh;
    background: radial-gradient(circle at 50% 50%, #6d5a9e, transparent 68%);
    animation: blobC 52s ease-in-out infinite alternate;
    opacity: 0.18;
}

@keyframes blobA { to { transform: translate(9vw, 12vh)  scale(1.15); } }
@keyframes blobB { to { transform: translate(-11vw, -8vh) scale(1.10); } }
@keyframes blobC { to { transform: translate(-7vw, -14vh) scale(1.18); } }

/* ==========================================================================
   THE GLASS MATERIAL — 3 layers on one element (reference recipe):
     ::before z -1  bend  (backdrop blur + SVG displacement → refraction)
     element        face  (tint + backdrop blur + drop shadow)
     ::after  z 2   edge  (white inset glossy highlights)
   ========================================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative;
    isolation: isolate;
    background: var(--ts-tint);
    border: none !important;
    border-radius: var(--ts-radius) !important;
    backdrop-filter: blur(10px) saturate(1.5);
    -webkit-backdrop-filter: blur(10px) saturate(1.5);
    box-shadow: var(--ts-shadow);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeUp 0.5s ease both;
    overflow: hidden;
}

/* bend layer — warps only what is painted below it (the background showing
   through the card), never the text, because it sits at z-index:-1. */
[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    backdrop-filter: blur(3px);
    -webkit-backdrop-filter: blur(3px);
    filter: url(#tsGlass);
    z-index: -1;
    pointer-events: none;
}

/* edge layer — the signature glossy rim. */
[data-testid="stVerticalBlockBorderWrapper"]::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    box-shadow:
        inset 2px 2px 2px var(--ts-edge-a),
        inset -2px -2px 2px var(--ts-edge-b),
        inset 0 1px 0 rgba(255, 255, 255, 0.30);
    z-index: 2;
    pointer-events: none;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px);
    box-shadow: var(--ts-shadow-hover);
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- Sidebar glass ---------- */

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.035);
    backdrop-filter: blur(22px) saturate(1.4);
    -webkit-backdrop-filter: blur(22px) saturate(1.4);
    border-right: 1px solid rgba(255, 255, 255, 0.10);
    box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] .block-container { color: var(--ts-text); }

/* ---------- Buttons: frosted pills ---------- */

.stButton > button, .stLinkButton a, a.st-link-button {
    background: rgba(255, 255, 255, 0.10) !important;
    color: var(--ts-text) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow:
        inset 1px 1px 1px rgba(255, 255, 255, 0.32),
        inset -1px -1px 1px rgba(255, 255, 255, 0.12),
        0 4px 12px rgba(8, 10, 24, 0.16);
    transition: all 0.2s ease !important;
}

.stButton > button:hover, .stLinkButton a:hover {
    transform: translateY(-1px);
    background: rgba(255, 255, 255, 0.16) !important;
}

.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.22), rgba(255, 255, 255, 0.02)),
        color-mix(in srgb, var(--primary, #6366f1) 78%, black) !important;
    border: 1px solid rgba(255, 255, 255, 0.34) !important;
    color: #ffffff !important;
    box-shadow:
        inset 1px 1px 1px rgba(255, 255, 255, 0.50),
        0 8px 24px color-mix(in srgb, var(--primary, #6366f1) 35%, transparent);
}

/* ---------- Metrics: same 3-layer glass as the verdict card ---------- */

[data-testid="stMetric"] {
    position: relative;
    isolation: isolate;
    background: var(--ts-tint);
    border: none;
    border-radius: 18px;
    padding: 12px 16px;
    backdrop-filter: blur(10px) saturate(1.5);
    -webkit-backdrop-filter: blur(10px) saturate(1.5);
    box-shadow: var(--ts-shadow);
    overflow: hidden;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    backdrop-filter: blur(3px);
    -webkit-backdrop-filter: blur(3px);
    filter: url(#tsGlass);
    z-index: -1;
    pointer-events: none;
}

[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    box-shadow:
        inset 1px 1px 2px var(--ts-edge-a),
        inset -1px -1px 1px var(--ts-edge-b);
    z-index: 2;
    pointer-events: none;
}

[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif;
    font-size: 1.35rem;
    color: var(--ts-text);
}

[data-testid="stMetricLabel"] { color: var(--ts-text-muted); font-size: 0.8rem; }

/* ---------- Status / expanders / alerts ---------- */

[data-testid="stStatus"], [data-testid="stExpander"] {
    background: var(--ts-tint) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(14px) saturate(1.5);
    -webkit-backdrop-filter: blur(14px) saturate(1.5);
    box-shadow:
        inset 1px 1px 1px rgba(255, 255, 255, 0.28),
        0 6px 18px rgba(8, 10, 24, 0.16);
}

[data-testid="stExpander"] details { background: transparent !important; }

[data-testid="stAlert"] { border-radius: 16px; }

[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.20);
    overflow: hidden;
    background: var(--ts-tint);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

hr { border-color: rgba(255, 255, 255, 0.16) !important; }

/* ---------- Inputs: frosted ---------- */

[data-testid="stTextInput"] input, .stTextArea textarea {
    background: rgba(255, 255, 255, 0.09) !important;
    color: var(--ts-text) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow:
        inset 1px 1px 1px rgba(255, 255, 255, 0.28),
        0 4px 12px rgba(8, 10, 24, 0.14);
}

[data-testid="stTextInput"] input::placeholder, .stTextArea textarea::placeholder {
    color: var(--ts-text-muted) !important;
}

[data-testid="stTextInput"] input:focus, .stTextArea textarea:focus {
    border-color: var(--ts-accent) !important;
    box-shadow:
        inset 1px 1px 1px rgba(255, 255, 255, 0.35),
        0 0 0 3px color-mix(in srgb, var(--primary, #6366f1) 22%, transparent) !important;
}

[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] span {
    color: var(--ts-text) !important;
}

/* Toggle: keep label solid and the knob clearly visible on glass */
[data-testid="stToggle"] label, [data-testid="stToggle"] span,
[data-testid="stToggle"] p {
    color: var(--ts-text) !important;
    font-weight: 600;
}

[data-testid="stToggle"] [role="switch"] {
    border: 1px solid rgba(255, 255, 255, 0.30);
    box-shadow: inset 1px 1px 1px rgba(255, 255, 255, 0.25);
}

[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background: color-mix(in srgb, var(--primary, #6366f1) 70%, transparent);
    border-color: rgba(255, 255, 255, 0.38);
}

/* ---------- Stance badges & chips ---------- */

.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.18rem 0.6rem;
    margin-left: 0.5rem;
    border-radius: 999px;
    vertical-align: middle;
    backdrop-filter: blur(6px);
    box-shadow: inset 1px 1px 1px rgba(255, 255, 255, 0.25);
}

.badge-supports    { background: rgba(34, 197, 94, 0.16);  color: var(--ts-pos); border: 1px solid rgba(34, 197, 94, 0.38); }
.badge-contradicts { background: rgba(239, 68, 68, 0.15);  color: var(--ts-neg); border: 1px solid rgba(239, 68, 68, 0.38); }
.badge-neutral     { background: rgba(120, 130, 155, 0.18); color: var(--ts-text-muted); border: 1px solid rgba(120, 130, 155, 0.32); }

.chip {
    display: inline-block;
    font-size: 0.72rem;
    padding: 0.12rem 0.55rem;
    margin-right: 0.35rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.26);
    box-shadow: inset 1px 1px 1px rgba(255, 255, 255, 0.28);
    color: var(--ts-text-muted);
}

/* ---------- Verdict banner: floating glass panel ---------- */

.verdict-banner {
    position: relative;
    isolation: isolate;
    border-radius: 26px;
    padding: 1.3rem 1.6rem;
    margin: 0.6rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    background: var(--ts-tint);
    backdrop-filter: blur(14px) saturate(1.6);
    -webkit-backdrop-filter: blur(14px) saturate(1.6);
    box-shadow: var(--ts-shadow);
    animation: fadeUp 0.55s ease both;
}

/* bend + edge layers, same recipe as cards */
.verdict-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    backdrop-filter: blur(3px);
    -webkit-backdrop-filter: blur(3px);
    filter: url(#tsGlass);
    z-index: -1;
    pointer-events: none;
}

.verdict-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    box-shadow:
        inset 2px 2px 2px var(--ts-edge-a),
        inset -2px -2px 2px var(--ts-edge-b),
        inset 0 1px 0 rgba(255, 255, 255, 0.30);
    z-index: 2;
    pointer-events: none;
}

.verdict-banner.real    { border: 1px solid rgba(34, 197, 94, 0.20); box-shadow: inset 2px 2px 2px var(--ts-edge-a), inset -2px -2px 2px var(--ts-edge-b), 0 10px 26px rgba(8, 10, 24, 0.16), 0 0 44px rgba(34, 197, 94, 0.10); }
.verdict-banner.fake    { border: 1px solid rgba(239, 68, 68, 0.20); box-shadow: inset 2px 2px 2px var(--ts-edge-a), inset -2px -2px 2px var(--ts-edge-b), 0 10px 26px rgba(8, 10, 24, 0.16), 0 0 44px rgba(239, 68, 68, 0.10); }
.verdict-banner.inconcl { border: 1px solid rgba(245, 158, 11, 0.20); box-shadow: inset 2px 2px 2px var(--ts-edge-a), inset -2px -2px 2px var(--ts-edge-b), 0 10px 26px rgba(8, 10, 24, 0.16), 0 0 44px rgba(245, 158, 11, 0.08); }

.verdict-banner .v-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    margin: 0;
}

.verdict-banner.real    .v-title { color: var(--ts-pos); }
.verdict-banner.fake    .v-title { color: var(--ts-neg); }
.verdict-banner.inconcl .v-title { color: var(--ts-amb); }

.verdict-banner .v-sub { color: var(--ts-text-muted); font-size: 0.9rem; margin: 0.25rem 0 0 0; }

/* ---------- Confidence ring ---------- */

@property --ring-progress { syntax: '<number>'; inherits: false; initial-value: 0; }

.ring {
    --ring-progress: 0;
    --ring-color: var(--ts-accent);
    position: relative;
    width: 108px;
    height: 108px;
    border-radius: 50%;
    background: conic-gradient(var(--ring-color) calc(var(--ring-progress) * 1%), var(--ts-ring-track) 0);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    animation: ringFill 1.3s cubic-bezier(0.2, 0.7, 0.3, 1) forwards;
    filter: drop-shadow(0 6px 14px rgba(8, 10, 24, 0.28));
}

.ring::after {
    content: '';
    position: absolute;
    width: 88px;
    height: 88px;
    border-radius: 50%;
    background: var(--ts-ring-inner);
    box-shadow:
        inset 0 2px 4px rgba(8, 10, 24, 0.18),
        inset 0 -1px 0 rgba(255, 255, 255, 0.14);
}

.ring .ring-label {
    position: relative;
    z-index: 1;
    text-align: center;
    font-family: 'Sora', sans-serif;
    color: var(--ts-text);
    line-height: 1.25;
}

.ring .ring-label .pct { font-size: 1.15rem; font-weight: 700; }
.ring .ring-label .txt { font-size: 0.62rem; color: var(--ts-text-muted); letter-spacing: 0.05em; }

@keyframes ringFill { from { --ring-progress: 0; } }

/* ---------- Evidence quote ---------- */

.evidence-quote {
    border-left: 3px solid var(--ts-accent);
    background: rgba(255, 255, 255, 0.08);
    padding: 0.55rem 0.85rem;
    border-radius: 0 12px 12px 0;
    color: var(--ts-text-muted);
    font-size: 0.86rem;
    font-style: italic;
    margin: 0.4rem 0;
}

/* ---------- Source card stagger ---------- */

.stagger { animation: fadeUp 0.5s ease both; }

/* ---------- Scrollbar ---------- */

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.16);
    border-radius: 8px;
}

/* ---------- Accessibility ---------- */

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


def inject_theme() -> None:
    """Apply the shared liquid-glass theme to the current page."""

    st.markdown(f"<style>{_THEME_CSS}</style>", unsafe_allow_html=True)
    st.markdown(_GLASS_SVG, unsafe_allow_html=True)
    st.markdown(_AMBIENT_HTML, unsafe_allow_html=True)


def render_html(block: str) -> None:
    """Render an HTML block safely from an indented source string.

    The block is dedented and collapsed onto a single line: newlines inside
    st.markdown HTML can terminate the HTML block (a whitespace-only line
    from an empty interpolation reads as blank) and turn following tags into
    visible text. A single line makes that failure impossible.
    """

    single_line = " ".join(dedent(block).split())
    st.markdown(single_line, unsafe_allow_html=True)


_STANCE_BADGES = {
    "SUPPORTS": ("badge-supports", "🟢 SUPPORTS"),
    "CONTRADICTS": ("badge-contradicts", "🔴 CONTRADICTS"),
    "NEUTRAL": ("badge-neutral", "⚪ NEUTRAL"),
}


def stance_badge_html(stance: str) -> str:
    """Return an HTML pill describing a source's stance."""

    css_class, label = _STANCE_BADGES.get(stance, _STANCE_BADGES["NEUTRAL"])
    return f"<span class='badge {css_class}'>{label}</span>"


_RING_COLORS = {
    "real": "var(--ts-pos)",
    "fake": "var(--ts-neg)",
    "inconcl": "var(--ts-amb)",
}


def verdict_ring_html(confidence: float, caption: str, kind: str) -> str:
    """Return an animated conic-gradient confidence ring."""

    percent = round(max(0.0, min(confidence, 1.0)) * 100)
    color = _RING_COLORS.get(kind, "var(--ts-accent)")
    return (
        f"<div class='ring' style='--ring-progress:{percent}; --ring-color:{color};'>"
        f"<div class='ring-label'><div class='pct'>{percent}%</div>"
        f"<div class='txt'>{caption}</div></div></div>"
    )


def stagger_delay(index: int) -> str:
    """Return an inline animation-delay style so cards enter one by one."""

    return f"animation-delay:{min(index, 8) * 0.07}s;"
