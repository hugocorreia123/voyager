"""Voyager — Agent Topology Evaluation — visual theme.

Palette, type and signature backdrop for this project's live demo.
Streamlit only. Pair with .streamlit/config.toml (base widget theme).
"""

import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Manrope:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: .01em; }

.stApp {
  background:
    url('data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%22420%22%20height%3D%22380%22%3E%3Cg%20stroke%3D%22%23E8B931%22%20stroke-opacity%3D%220.10%22%20fill%3D%22none%22%3E%3Cpath%20d%3D%22M40%2060%20L100%2080%20L160%2060%20L220%2085%22/%3E%3Cpath%20d%3D%22M320%2090%20L360%2060%20M320%2090%20L370%20100%20M320%2090%20L340%20140%20M320%2090%20L280%20130%22/%3E%3Cpath%20d%3D%22M120%20280%20L170%20240%20L220%20280%20M170%20240%20L170%20190%22/%3E%3C/g%3E%3Cg%20fill%3D%22%23F3E3B2%22%20fill-opacity%3D%220.16%22%3E%3Ccircle%20cx%3D%2240%22%20cy%3D%2260%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%22100%22%20cy%3D%2280%22%20r%3D%223%22/%3E%3Ccircle%20cx%3D%22160%22%20cy%3D%2260%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%22220%22%20cy%3D%2285%22%20r%3D%223%22/%3E%3Ccircle%20cx%3D%22320%22%20cy%3D%2290%22%20r%3D%223.5%22/%3E%3Ccircle%20cx%3D%22360%22%20cy%3D%2260%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%22370%22%20cy%3D%22100%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%22340%22%20cy%3D%22140%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%22280%22%20cy%3D%22130%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%22120%22%20cy%3D%22280%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%22170%22%20cy%3D%22240%22%20r%3D%223%22/%3E%3Ccircle%20cx%3D%22220%22%20cy%3D%22280%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%22170%22%20cy%3D%22190%22%20r%3D%222.5%22/%3E%3Ccircle%20cx%3D%2260%22%20cy%3D%22330%22%20r%3D%221.6%22/%3E%3Ccircle%20cx%3D%22390%22%20cy%3D%22300%22%20r%3D%221.6%22/%3E%3Ccircle%20cx%3D%22250%22%20cy%3D%2230%22%20r%3D%221.6%22/%3E%3C/g%3E%3C/svg%3E') repeat,
    radial-gradient(1100px 520px at 85% -12%, rgba(139,124,246,0.12), transparent 60%),radial-gradient(860px 480px at -10% 112%, rgba(232,185,49,0.07), transparent 55%),linear-gradient(180deg, #0B1026 0%, #090D1F 100%);
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }

/* hero */
.tr-hero {
  border-radius: 18px;
  padding: 26px 30px 24px 30px;
  margin: 4px 0 14px 0;
  background: linear-gradient(135deg, rgba(21,27,59,0.94) 0%, rgba(11,16,38,0.94) 70%);
  border: 1px solid #E8B93140;
  box-shadow: 0 12px 40px -18px #E8B93159;
}
.tr-hero .eyebrow {
  font-family: 'Manrope', sans-serif;
  font-size: .72rem; font-weight: 700; letter-spacing: .22em;
  text-transform: uppercase; color: #E8B931; margin-bottom: 6px;
}
.tr-hero h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(1.7rem, 3.2vw, 2.5rem); font-weight: 800;
  margin: 0 0 8px 0; padding: 0; color: #EEF0FA; line-height: 1.08;
}
.tr-hero .tag { color: #A7ACCB; font-size: 1.0rem; max-width: 72ch; margin: 0; }
.tr-hero .meta { color: #A7ACCB; opacity: .8; font-size: .8rem; margin-top: 10px; letter-spacing: .04em; }

/* metric cards */
[data-testid="stMetric"] {
  background: rgba(21, 27, 59, 0.58);
  border: 1px solid #ffffff1a;
  border-left: 3px solid #E8B931;
  border-radius: 14px;
  padding: 14px 16px 12px 16px;
}
[data-testid="stMetricLabel"] p {
  text-transform: uppercase; letter-spacing: .07em;
  font-size: .74rem; font-weight: 700; color: #A7ACCB;
}
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: #EEF0FA; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #ffffff1a; }
.stTabs [data-baseweb="tab"] {
  padding: 10px 16px; font-weight: 600; border-radius: 10px 10px 0 0;
}
.stTabs [aria-selected="true"] {
  color: #E8B931 !important;
  box-shadow: inset 0 -2px 0 #E8B931;
  background: #E8B93114;
}

/* buttons */
.stButton > button { border-radius: 12px; font-weight: 600; }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #E8B931 0%, #B9891C 100%);
  color: #171102; border: 0;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  filter: brightness(1.08);
}

/* containers */
[data-testid="stExpander"] {
  border: 1px solid #ffffff1a; border-radius: 12px; background: rgba(21, 27, 59, 0.58);
}
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #ffffff1a; }
[data-testid="stCaptionContainer"], .stCaption { color: #A7ACCB; }
[data-testid="stSidebar"] { background: #0B1026; border-right: 1px solid #ffffff1a; }
hr { border-color: #ffffff1a; }
[data-testid="stDataFrame"] { border: 1px solid #ffffff1a; border-radius: 12px; }

/* ---------- motion layer: Apple-quiet, minimal ---------- */
html { scroll-behavior: smooth; }

.tr-hero, [data-testid="stMetric"], [data-testid="stExpander"] {
  backdrop-filter: blur(12px) saturate(1.15);
  -webkit-backdrop-filter: blur(12px) saturate(1.15);
}

[data-testid="stMetric"], .stButton > button, [data-testid="stExpander"] {
  transition: transform .28s cubic-bezier(.22,.61,.36,1),
              box-shadow .28s cubic-bezier(.22,.61,.36,1),
              border-color .28s ease, filter .2s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: #E8B93166;
  box-shadow: 0 16px 38px -18px #E8B93159;
}
.stButton > button:hover { transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0) scale(.99); }

@media (prefers-reduced-motion: no-preference) {
  @keyframes tr-rise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: none; }
  }
  @keyframes tr-fade { from { opacity: 0; } to { opacity: 1; } }

  .tr-hero { animation: tr-rise .7s cubic-bezier(.22,.61,.36,1) both; }
  [data-testid="stMetric"] { animation: tr-rise .6s cubic-bezier(.22,.61,.36,1) both; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] { animation-delay: .06s; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] { animation-delay: .14s; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] { animation-delay: .22s; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] { animation-delay: .30s; }
  .stTabs { animation: tr-fade .5s ease-out both; animation-delay: .15s; }

  @supports (animation-timeline: view()) {
    [data-testid="stPlotlyChart"], [data-testid="stImage"],
    [data-testid="stExpander"], [data-testid="stDataFrame"] {
      animation: tr-rise .7s cubic-bezier(.22,.61,.36,1) both;
      animation-timeline: view();
      animation-range: entry 0% entry 38%;
    }
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


def inject() -> None:
    """Apply the theme. Call once, right after st.set_page_config."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def hero(eyebrow: str, title: str, tag: str, meta: str = "") -> None:
    """The styled header banner. Replaces st.title + st.caption."""
    meta_html = f'<div class="meta">{meta}</div>' if meta else ""
    st.markdown(
        f'''<div class="tr-hero">
  <div class="eyebrow">{eyebrow}</div>
  <h1>{title}</h1>
  <p class="tag">{tag}</p>
  {meta_html}
</div>''',
        unsafe_allow_html=True,
    )
