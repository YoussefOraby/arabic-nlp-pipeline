import json
import subprocess
import sys
import sqlite3
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, ctx, dcc, html, Input, Output, State, no_update
from dash.dash_table import DataTable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pipeline.db"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PROGRESS_FILE = PROJECT_ROOT / "data" / ".pipeline_progress.json"
PYTHON = sys.executable

SCRIPT_ORDER = ["clean_data.py", "sentiment.py", "ner.py", "store_db.py"]
STEPS = [
    {"id": "collect", "label": "Collecting comments"},
    {"id": "clean", "label": "Cleaning text"},
    {"id": "sentiment", "label": "Analyzing sentiment"},
    {"id": "ner", "label": "Extracting entities"},
    {"id": "store", "label": "Saving results"},
]

SENTIMENT_COLORS = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#6b7280"}
STYLE = {
    "bg": "#f0f2f5",
    "card": "#ffffff",
    "accent": "#2563eb",
    "accent-hover": "#1d4ed8",
    "text": "#1e293b",
    "text-muted": "#64748b",
    "border": "#e2e8f0",
    "success": "#10b981",
    "danger": "#ef4444",
    "shadow": "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)",
    "shadow-lg": "0 4px 6px rgba(0,0,0,0.07), 0 10px 15px rgba(0,0,0,0.05)",
    "radius": "12px",
    "font": "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
}

card_style = {
    "backgroundColor": STYLE["card"],
    "borderRadius": STYLE["radius"],
    "boxShadow": STYLE["shadow"],
    "padding": "24px",
    "marginBottom": "24px",
}

_empty_msg = "No data yet \u2014 run an analysis above"
EMPTY_FIG = px.pie(title="Sentiment Distribution").update_layout(
    font_family=STYLE["font"], title_font_size=18,
    annotations=[dict(text=_empty_msg, showarrow=False, font=dict(size=14, color=STYLE["text-muted"]))],
)
EMPTY_BAR_FIG = px.bar(title="Top 10 Entities").update_layout(
    font_family=STYLE["font"], title_font_size=18,
    annotations=[dict(text=_empty_msg, showarrow=False, font=dict(size=14, color=STYLE["text-muted"]))],
)


# ─── HELPERS ─────────────────────────────────────────────────────────

def load_data():
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql("SELECT * FROM posts", conn)
        conn.close()
        return df if not df.empty else None
    except Exception:
        return None


def build_pie(df):
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    return px.pie(
        counts,
        names="sentiment",
        values="count",
        title="Sentiment Distribution",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    ).update_layout(font_family=STYLE["font"], title_font_size=18)


def build_bar(df):
    counter = Counter()
    for row in df["entities"]:
        try:
            for entity_text, _ in json.loads(row):
                counter[entity_text] += 1
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    top = counter.most_common(10)
    edf = pd.DataFrame(top, columns=["entity", "count"])
    fig = px.bar(edf, x="count", y="entity", orientation="h", title="Top 10 Entities")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        font_family=STYLE["font"],
        title_font_size=18,
    )
    return fig


def build_table(df, sentiment_filter, search_text):
    filtered = df.copy()
    if sentiment_filter and sentiment_filter != "all":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]
    if search_text:
        filtered = filtered[
            filtered["clean_text"].str.contains(search_text, case=False, na=False)
        ]
    cols = ["clean_text", "sentiment", "confidence"]
    return filtered.tail(20)[cols].to_dict("records")


# ─── PROGRESS TRACKER ────────────────────────────────────────────────

def run_pipeline_thread(topic):
    start = time.time()
    scripts = ["load_live_data.py"] + SCRIPT_ORDER

    for i, script in enumerate(scripts):
        try:
            payload = {
                "status": "running",
                "step": i,
                "script": script,
                "start_time": start,
                "error": "",
                "stdout": "",
                "stderr": "",
            }
            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            cmd = [PYTHON, str(SCRIPTS_DIR / script)]
            if script == "load_live_data.py":
                cmd.append(topic)

            result = subprocess.run(
                cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600,
            )

            if result.returncode != 0:
                payload["status"] = "failed"
                payload["error"] = result.stderr.strip() or result.stdout.strip()
                payload["stdout"] = result.stdout.strip()
                payload["stderr"] = result.stderr.strip()
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                return

        except Exception as e:
            payload = {
                "status": "failed",
                "step": i,
                "script": script,
                "start_time": start,
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
            }
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return

    payload = {
        "status": "done",
        "step": len(scripts),
        "script": "",
        "start_time": start,
        "error": "",
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def format_elapsed(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def build_stepper(progress):
    status = progress.get("status", "")
    step = progress.get("step", 0)
    error = progress.get("error", "")
    start_time = progress.get("start_time", 0)
    elapsed = time.time() - start_time if start_time else 0

    def step_class(i):
        if status == "done":
            return "done"
        if status == "failed":
            return "failed" if i == step else ("done" if i < step else "pending")
        return "active" if i == step else ("done" if i < step else "pending")

    icons = {"done": "\u2713", "failed": "\u2717", "active": "\u25B6", "pending": "\u25CB"}
    colors = {"done": STYLE["success"], "failed": STYLE["danger"], "active": STYLE["accent"], "pending": STYLE["text-muted"]}
    bg = {"done": "#ecfdf5", "failed": "#fef2f2", "active": "#eff6ff", "pending": "#f8fafc"}

    items = []
    for i, s in enumerate(STEPS):
        cls = step_class(i)
        c = colors[cls]

        if i > 0:
            prev_cls = step_class(i - 1)
            arrow_color = colors[prev_cls] if prev_cls == "done" or (status == "running" and i - 1 < step) else STYLE["border"]
            items.append(html.Span("\u2192", style={"color": arrow_color, "fontSize": "20px", "margin": "0 4px", "lineHeight": "40px"}))

        items.append(
            html.Div(
                style={"display": "flex", "flexDirection": "column", "alignItems": "center", "minWidth": "120px"},
                children=[
                    html.Div(
                        icons[cls],
                        style={
                            "width": "40px", "height": "40px", "borderRadius": "50%",
                            "backgroundColor": bg[cls], "border": f"2px solid {c}",
                            "display": "flex", "alignItems": "center", "justifyContent": "center",
                            "fontSize": "16px", "fontWeight": "700", "color": c,
                        },
                    ),
                    html.Div(
                        s["label"],
                        style={
                            "marginTop": "8px", "fontSize": "13px",
                            "fontWeight": "600" if cls == "active" else "400",
                            "color": c, "textAlign": "center", "whiteSpace": "nowrap",
                        },
                    ),
                    html.Div(
                        f"\u23F3 Running\u2026 ({format_elapsed(elapsed)})",
                        style={"fontSize": "11px", "color": STYLE["accent"], "marginTop": "4px"},
                    ) if cls == "active" else None,
                    html.Div(
                        error[:120] + ("..." if len(error) > 120 else ""),
                        style={"fontSize": "11px", "color": STYLE["danger"], "textAlign": "center", "maxWidth": "160px", "marginTop": "4px", "wordBreak": "break-word"},
                    ) if cls == "failed" and error else None,
                ],
            )
        )

    return html.Div(
        style={"display": "flex", "alignItems": "flex-start", "justifyContent": "center", "padding": "24px 8px", "flexWrap": "wrap", "gap": "0"},
        children=items,
    )


# ─── APP ─────────────────────────────────────────────────────────────

app = Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": STYLE["bg"],
        "fontFamily": STYLE["font"],
        "minHeight": "100vh",
        "margin": "0",
        "padding": "0",
    },
    children=[
        # ─── HEADER ───────────────────────────────────────────────
        html.Div(
            style={
                "backgroundColor": STYLE["card"],
                "borderBottom": f"1px solid {STYLE['border']}",
                "boxShadow": STYLE["shadow"],
                "padding": "40px 20px 32px",
            },
            children=[
                html.H1(
                    "Arabic NLP Sentiment Dashboard",
                    style={
                        "textAlign": "center",
                        "color": STYLE["text"],
                        "fontSize": "32px",
                        "fontWeight": "700",
                        "margin": "0 0 28px 0",
                        "letterSpacing": "-0.5px",
                    },
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "gap": "12px",
                        "maxWidth": "900px",
                        "margin": "0 auto",
                    },
                    children=[
                        dcc.Input(
                            id="topic-input",
                            type="text",
                            placeholder="Enter any topic to analyze (brand, person, event, hashtag\u2026)",
                            style={
                                "flex": "1",
                                "padding": "16px 20px",
                                "fontSize": "16px",
                                "lineHeight": "1.5",
                                "boxSizing": "border-box",
                                "borderRadius": "8px",
                                "border": f"1.5px solid {STYLE['border']}",
                                "outline": "none",
                                "transition": "border-color 0.2s",
                            },
                        ),
                        html.Button(
                            "Analyze",
                            id="analyze-btn",
                            n_clicks=0,
                            style={
                                "padding": "16px 32px",
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "cursor": "pointer",
                                "borderRadius": "8px",
                                "border": "none",
                                "backgroundColor": STYLE["accent"],
                                "color": "#fff",
                                "transition": "background-color 0.2s, transform 0.1s",
                            },
                        ),
                    ],
                ),
                html.Div(
                    id="pipeline-status",
                    style={
                        "textAlign": "center",
                        "marginTop": "12px",
                        "fontSize": "14px",
                        "color": STYLE["text-muted"],
                        "minHeight": "20px",
                    },
                ),
            ],
        ),
        # ─── MAIN CONTENT ─────────────────────────────────────────
        html.Div(
            style={"maxWidth": "1200px", "margin": "0 auto", "padding": "32px 20px"},
            children=[
                # Empty state
                html.Div(
                    id="empty-state",
                    children=[
                        html.Div(
                            style={**card_style, "textAlign": "center", "padding": "80px 40px"},
                            children=[
                                html.Div("\U0001f50d", style={"fontSize": "48px", "marginBottom": "16px"}),
                                html.H2(
                                    "Search a topic above to see live sentiment analysis",
                                    style={"color": STYLE["text"], "fontSize": "22px", "fontWeight": "600", "margin": "0 0 8px 0"},
                                ),
                                html.P(
                                    "Results will appear here after analysis completes.",
                                    style={"color": STYLE["text-muted"], "fontSize": "15px", "margin": "0"},
                                ),
                            ],
                        )
                    ],
                ),
                # Progress stepper (hidden initially)
                html.Div(
                    id="progress-section",
                    style={"display": "none"},
                    children=[
                        html.Div(
                            style=card_style,
                            children=[
                                html.H3(
                                    "Pipeline Progress",
                                    style={"color": STYLE["text"], "fontSize": "18px", "fontWeight": "600", "margin": "0 0 4px 0", "textAlign": "center"},
                                ),
                                html.Div(id="progress-stepper"),
                            ],
                        ),
                        # Interval for polling progress
                        dcc.Interval(
                            id="progress-interval",
                            interval=1500,
                            disabled=True,
                        ),
                    ],
                ),
                # Results section (hidden initially)
                html.Div(
                    id="results-section",
                    style={"display": "none"},
                    children=[
                        html.H3(
                            id="current-topic",
                            style={
                                "textAlign": "center",
                                "color": STYLE["text-muted"],
                                "fontSize": "16px",
                                "fontWeight": "400",
                                "margin": "0 0 20px 0",
                            },
                        ),
                        html.Div(
                            style=card_style,
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "center", "gap": "32px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                            children=[
                                                html.Label("Sentiment:", style={"fontWeight": "600", "color": STYLE["text"], "fontSize": "14px"}),
                                                dcc.Dropdown(
                                                    id="sentiment-filter",
                                                    options=[
                                                        {"label": "All", "value": "all"},
                                                        {"label": "Positive", "value": "positive"},
                                                        {"label": "Negative", "value": "negative"},
                                                        {"label": "Neutral", "value": "neutral"},
                                                    ],
                                                    value="all",
                                                    clearable=False,
                                                    style={"width": "180px"},
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                            children=[
                                                html.Label("Search:", style={"fontWeight": "600", "color": STYLE["text"], "fontSize": "14px"}),
                                                dcc.Input(
                                                    id="search-box",
                                                    type="text",
                                                    placeholder="Filter by text\u2026",
                                                    style={
                                                        "padding": "8px 12px",
                                                        "fontSize": "14px",
                                                        "lineHeight": "1.5",
                                                        "boxSizing": "border-box",
                                                        "width": "260px",
                                                        "borderRadius": "6px",
                                                        "border": f"1px solid {STYLE['border']}",
                                                        "outline": "none",
                                                    },
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            color=STYLE["accent"],
                            children=[
                                html.Div(style=card_style, children=[dcc.Graph(id="pie-chart")]),
                                html.Div(style=card_style, children=[dcc.Graph(id="bar-chart")]),
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.H2("Recent Posts", style={"color": STYLE["text"], "fontSize": "20px", "fontWeight": "600", "margin": "0 0 16px 0"}),
                                        DataTable(
                                            id="data-table",
                                            columns=[
                                                {"name": "Text", "id": "clean_text"},
                                                {"name": "Sentiment", "id": "sentiment"},
                                                {"name": "Confidence", "id": "confidence", "type": "numeric", "format": {"specifier": ".4f"}},
                                            ],
                                            page_size=20,
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "textAlign": "left",
                                                "maxWidth": "420px",
                                                "overflow": "hidden",
                                                "textOverflow": "ellipsis",
                                                "fontFamily": STYLE["font"],
                                                "fontSize": "14px",
                                            },
                                            style_header={"fontWeight": "600", "backgroundColor": STYLE["bg"], "color": STYLE["text"]},
                                            style_data_conditional=[
                                                {"if": {"filter_query": "{sentiment} = 'positive'"}, "backgroundColor": "#ecfdf5"},
                                                {"if": {"filter_query": "{sentiment} = 'negative'"}, "backgroundColor": "#fef2f2"},
                                                {"if": {"filter_query": "{sentiment} = 'neutral'"}, "backgroundColor": "#f8fafc"},
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Store(id="pipeline-running", data=False),
    ],
)


# ═══════════════════════════════════════════════════════════════════
# SINGLE CONSOLIDATED CALLBACK
# ═══════════════════════════════════════════════════════════════════
@app.callback(
    Output("progress-stepper", "children"),
    Output("pipeline-status", "children"),
    Output("pipeline-running", "data"),
    Output("progress-interval", "disabled"),
    Output("empty-state", "style"),
    Output("progress-section", "style"),
    Output("results-section", "style"),
    Output("pie-chart", "figure"),
    Output("bar-chart", "figure"),
    Output("data-table", "data"),
    Input("analyze-btn", "n_clicks"),
    Input("progress-interval", "n_intervals"),
    Input("sentiment-filter", "value"),
    Input("search-box", "value"),
    State("topic-input", "value"),
    State("pipeline-running", "data"),
    prevent_initial_call=True,
)
def dashboard_callback(analyze_clicks, n_intervals, sentiment_filter, search_text, topic, running):
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # ── Analyze button clicked: start pipeline ──
    if triggered_id == "analyze-btn":
        err_el = html.Span(
            "Please enter a topic before analyzing.",
            style={"color": STYLE["danger"], "fontWeight": "500"},
        )
        if not topic or not topic.strip():
            return no_update, err_el, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        running_el = html.Span(
            "Pipeline already running \u2014 wait for it to finish.",
            style={"color": STYLE["danger"], "fontWeight": "500"},
        )
        if running:
            return no_update, running_el, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        topic = topic.strip()
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        thread = threading.Thread(target=run_pipeline_thread, args=(topic,), daemon=True)
        thread.start()
        return (
            "",
            html.Span(f'Analyzing: "{topic}" \u2026', style={"color": STYLE["accent"], "fontWeight": "500"}),
            True,
            False,  # enable interval
            {"display": "none"},
            {"display": "block"},
            {"display": "none"},
            EMPTY_FIG,
            EMPTY_BAR_FIG,
            [],
        )

    # ── Progress interval fired: poll pipeline progress ──
    if triggered_id == "progress-interval":
        if not running:
            return (no_update,) * 10

        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                progress = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return (no_update,) * 10

        stepper = build_stepper(progress)

        if progress["status"] == "failed":
            err_msg = progress.get("error", "Unknown error")
            stderr = progress.get("stderr", "")
            stdout = progress.get("stdout", "")
            full = err_msg
            if stderr and stderr != err_msg:
                full += f"\n\n[stderr]\n{stderr}"
            status_el = html.Div(
                [
                    html.Span(
                        f"Failed at step \u201c{STEPS[progress['step']]['label']}\u201d",
                        style={"color": STYLE["danger"], "fontWeight": "500"},
                    ),
                    html.Pre(
                        full,
                        style={
                            "color": STYLE["danger"], "fontSize": "12px",
                            "maxWidth": "800px", "margin": "8px auto 0",
                            "whiteSpace": "pre-wrap", "textAlign": "left",
                            "backgroundColor": "#fef2f2", "padding": "12px",
                            "borderRadius": "6px",
                        },
                    ),
                ],
                style={"textAlign": "center"},
            )
            return (
                stepper, status_el,
                False, True,
                {"display": "none"}, {"display": "block"}, {"display": "none"},
                EMPTY_FIG, EMPTY_BAR_FIG, [],
            )

        if progress["status"] == "done":
            df = load_data()
            if df is not None and not df.empty:
                status_el = html.Span(
                    f"Analysis complete \u2014 {len(df)} posts processed.",
                    style={"color": STYLE["success"], "fontWeight": "500"},
                )
                return (
                    stepper, status_el,
                    False, True,
                    {"display": "none"}, {"display": "none"}, {"display": "block"},
                    build_pie(df), build_bar(df),
                    build_table(df, sentiment_filter or "", search_text or ""),
                )
            status_el = html.Div(
                [
                    html.Span(
                        "Pipeline completed but no data was found.",
                        style={"color": STYLE["danger"], "fontWeight": "500"},
                    ),
                    html.P(
                        "Check that your YouTube API key is valid and try a different topic.",
                        style={"color": STYLE["text-muted"], "fontSize": "13px", "marginTop": "4px"},
                    ),
                ],
                style={"textAlign": "center"},
            )
            return (
                stepper, status_el,
                False, True,
                {"display": "none"}, {"display": "none"}, {"display": "none"},
                EMPTY_FIG, EMPTY_BAR_FIG, [],
            )

        # Still running
        return (
            stepper, no_update,
            True, False,
            {"display": "none"}, {"display": "block"}, {"display": "none"},
            EMPTY_FIG, EMPTY_BAR_FIG, [],
        )

    # ── Sentiment filter or search box changed ──
    df = load_data()
    if df is not None and not df.empty:
        return (
            no_update, no_update,
            no_update, no_update,
            {"display": "none"}, {"display": "none"}, {"display": "block"},
            build_pie(df), build_bar(df),
            build_table(df, sentiment_filter or "", search_text or ""),
        )

    return (
        no_update, no_update,
        no_update, no_update,
        {"display": "block"}, {"display": "none"}, {"display": "none"},
        EMPTY_FIG, EMPTY_BAR_FIG, [],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
