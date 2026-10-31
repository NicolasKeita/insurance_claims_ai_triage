"""Self-contained dark report document with no network dependencies."""

from .text import _text

_STYLE = """
:root{color-scheme:dark;--bg:#0e171c;--card:#172329;--accent:#1b3635;
--line:#34454c;--text:#edf5f3;--muted:#b0c2c3;--teal:#a5e1d5}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);
color:var(--text);font:15px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
a{color:var(--teal);text-underline-offset:3px}a:focus-visible,summary:focus-visible{
outline:3px solid #d4f0c6;outline-offset:5px}h1,h2,h3,p,figure,dl{margin:0}
h1{font-size:clamp(26px,4vw,38px);line-height:1.2;letter-spacing:-.025em}
h2{font-size:20px;line-height:1.4}h3{font-size:17px;line-height:1.4}
p+p{margin-top:10px}small,.muted{color:var(--muted)}small{font-size:12px}
.shell{width:min(1160px,calc(100% - 40px));margin:auto;padding:30px 0 46px}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:11px;color:var(--teal);
margin-bottom:9px}.header{margin-bottom:22px}.header>p{margin-top:10px}
.facts{display:flex;flex-wrap:wrap;gap:16px 32px;margin-top:22px}
dt{color:var(--muted);font-size:12px}dd{margin:2px 0 0;font-weight:600}
.notice{border-left:3px solid var(--teal);background:var(--accent);padding:10px 15px;
margin:22px 0}.navigation{display:flex;flex-wrap:wrap;gap:12px 24px;padding:13px 0;
border-block:1px solid var(--line);margin-bottom:22px;font-size:13px}
.hero{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(300px,1fr);gap:20px;
align-items:start}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:22px}.result{background:var(--accent)}
.result>p,.result>h3{margin-top:12px}.result .eyebrow{margin-top:22px;margin-bottom:0}
.button{display:inline-block;margin-top:20px;padding:9px 15px;border:1px solid var(--teal);
border-radius:6px;text-decoration:none;color:#102522;background:var(--teal);font-weight:650;
font-size:13px}.photo{overflow:hidden;padding:0}.photo-stage{position:relative;
line-height:0;background:#0b1318}.photo-stage>img{display:block;width:100%;height:auto}
.photo figcaption{padding:12px 17px;font-size:12px;color:var(--muted)}
.photo figcaption a{overflow-wrap:anywhere}.photo-placeholder{min-height:220px;
display:grid;place-content:center;padding:25px;text-align:center;color:var(--muted)}
.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.overlay path{fill:#e8b54338;stroke:#f4c963;stroke-width:.3;fill-rule:evenodd}
.overlay rect{fill:none;stroke:#f4c963;stroke-width:.4;vector-effect:non-scaling-stroke}
section[id]{scroll-margin-top:20px}.section{margin-top:26px}.section>h2{margin-bottom:13px}
.section>p{margin-bottom:13px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
gap:15px}.file{display:flex;flex-direction:column;gap:7px}.file h3{font-size:15px}
.file code{font-size:12px;overflow-wrap:anywhere}.badge{display:inline-block;
font-size:11px;padding:2px 7px;border:1px solid var(--line);border-radius:4px;
color:var(--muted)}.file .badge{align-self:flex-start}.recommendation>p{margin-top:14px}
ol,ul{padding-left:22px}li+li{margin-top:8px}.source-text{white-space:pre-wrap;
overflow-wrap:anywhere}.references{font-family:ui-monospace,monospace;font-size:11px;
color:var(--muted);overflow-wrap:anywhere}.recommendation h3{margin-top:20px}
details{margin-top:15px;border-top:1px solid var(--line);padding-top:13px}
summary{cursor:pointer;color:var(--teal);font-size:13px}details>div,details>p,
details>pre{margin-top:13px}pre{white-space:pre-wrap;word-break:break-word;
font-size:12px;max-height:550px;overflow:auto}.table-wrap{overflow:auto}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:10px;text-align:left;
vertical-align:top;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
th{color:var(--muted);font-weight:500}.footer{margin-top:32px;padding-top:15px;
border-top:1px solid var(--line);color:var(--muted);font-size:12px}.catalogue-card h2{
font-size:20px}.catalogue-card p{margin-top:10px}.catalogue-card code{overflow-wrap:anywhere}
@media(max-width:760px){.hero,.grid{grid-template-columns:1fr}.shell{width:calc(100% - 28px);
padding-top:22px}.hero>.result{grid-row:1}.card{padding:18px}.photo{padding:0}}
@media print{:root{color-scheme:light;--bg:white;--card:white;--accent:#eef6f2;
--line:#777;--text:#111;--muted:#444;--teal:#175e51}body{font-size:11px}.shell{width:100%;
padding:0}.navigation,.button{display:none}.card,figure{break-inside:avoid}
details{display:block}.photo-stage{background:white}a{color:#175e51}}
"""


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "img-src 'self' file: data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>{_text(title)}</title><style>{_STYLE}</style></head><body>"
        f'<main class="shell">{body}</main></body></html>'
    )
