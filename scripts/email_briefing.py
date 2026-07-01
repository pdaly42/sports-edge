#!/usr/bin/env python3
"""
Read today's predictions JSON and email a formatted NFL edge briefing.

Usage:
  python3 scripts/email_briefing.py [predictions_YYYY-MM-DD.json]

Env vars:
  GMAIL_APP_PASSWORD  — Gmail App Password for pdaly42@gmail.com
  TO_EMAIL            — Recipient email (default: pdaly42@gmail.com)
"""
from __future__ import annotations
import json, os, smtplib, sys
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

FROM_EMAIL = "pdaly42@gmail.com"
TO_EMAIL   = os.environ.get("TO_EMAIL", "pdaly42@gmail.com")

STRONG_COLOR   = "#1a7a4a"
MODERATE_COLOR = "#b86e00"
BLUE_COLOR     = "#3b82f6"
BORDER_COLOR   = "#e2e8f0"
MUTED_COLOR    = "#718096"
TEXT_COLOR     = "#1a202c"
BG_COLOR       = "#f8f9fa"


def format_odds(odds) -> str:
    if odds is None:
        return "N/A"
    odds = int(odds)
    return f"+{odds}" if odds > 0 else str(odds)


def time_et(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        offset = 5 if dt.month in (11, 12, 1, 2, 3) else 4
        et = dt.astimezone(timezone(timedelta(hours=-offset)))
        return et.strftime("%-I:%M %p ET")
    except Exception:
        return ""


def _game_row(g: dict) -> str:
    away    = g.get("away_team", "Away")
    home    = g.get("home_team", "Home")
    h_p     = g.get("model_home_prob")
    a_p     = g.get("model_away_prob")
    h_e     = g.get("home_edge") or 0
    a_e     = g.get("away_edge") or 0
    best    = max(h_e, a_e)
    flag    = " ★" if g.get("best_bet") else ""
    t       = time_et(g.get("commence_time", ""))

    color    = STRONG_COLOR if best >= 0.08 else (MODERATE_COLOR if best >= 0.03 else MUTED_COLOR)
    edge_str = f"+{round(best*100,1)}%" if best > 0 else "—"
    h_str    = f"{round(h_p*100)}%" if h_p is not None else "—"
    a_str    = f"{round(a_p*100)}%" if a_p is not None else "—"
    time_tag = (
        "<br>"
        f'<span style="font-size:11px;color:{MUTED_COLOR};">{t}</span>'
        if t else ""
    )

    return (
        f'<tr style="border-bottom:1px solid {BORDER_COLOR};">'
        f'<td style="padding:7px 4px;">{away} @ {home}{flag}{time_tag}</td>'
        f'<td align="right" style="padding:7px 4px;color:{MUTED_COLOR};">{h_str}</td>'
        f'<td align="right" style="padding:7px 4px;color:{MUTED_COLOR};">{a_str}</td>'
        f'<td align="right" style="padding:7px 4px;font-weight:700;color:{color};">{edge_str}</td>'
        "</tr>"
    )


def _bet_card(g: dict) -> str:
    bb     = g["best_bet"]
    side   = bb["side"]
    team   = g["home_team"] if side == "home" else g["away_team"]
    odds   = format_odds(bb.get("odds"))
    edge   = round(bb["edge"] * 100, 1)
    ev     = round((bb.get("ev") or 0) * 100, 1)
    kelly  = round((g.get(f"{side}_kelly") or 0) * 100, 1)
    strong = bb.get("strength") == "strong"
    color  = STRONG_COLOR if strong else MODERATE_COLOR
    label  = "STRONG EDGE" if strong else "MODERATE EDGE"
    t      = time_et(g.get("commence_time", ""))
    away   = g.get("away_team", "")
    home   = g.get("home_team", "")
    just   = g.get("justification", "")

    time_note = f" ({t})" if t else ""
    game_line = (
        f'<div style="font-size:13px;color:{MUTED_COLOR};margin-bottom:6px;">'
        f"{away} @ {home}{time_note}"
        "</div>"
    )

    just_block = (
        f'<p style="font-size:13px;color:{TEXT_COLOR};margin:0 0 6px;line-height:1.5;">'
        f"{just}</p>"
        if just else ""
    )

    # Injury callout
    h_qbi     = g.get("home_qb_injury_impact") or 0
    a_qbi     = g.get("away_qb_injury_impact") or 0
    max_qbi   = max(h_qbi, a_qbi)
    inj_block = ""
    if max_qbi >= 0.2:
        inj_team  = g["home_team"] if h_qbi >= a_qbi else g["away_team"]
        inj_level = "Out/Doubtful" if max_qbi >= 0.5 else "Questionable"
        inj_block = (
            f'<div style="margin-top:8px;padding:6px 10px;background:#fff7ed;'
            f"border-left:3px solid #f97316;border-radius:4px;font-size:12px;color:#9a3412;\">"
            f"⚠️ {inj_team} QB: {inj_level}</div>"
        )

    # O/U annotation
    ou_block = ""
    totals   = g.get("totals")
    if totals and totals.get("best_ou_bet"):
        oub     = totals["best_ou_bet"]
        ou_side = oub["side"].upper()
        ou_line = totals.get("market_line", "?")
        ou_pred = totals.get("predicted_total", "?")
        ou_edge = round((oub.get("edge") or 0) * 100, 1)
        ou_odds = format_odds(oub.get("odds"))
        ou_block = (
            f'<div style="margin-top:10px;padding:8px 12px;background:#eef6ff;'
            f"border-left:3px solid {BLUE_COLOR};border-radius:4px;font-size:13px;\">"
            f"<strong>Also:</strong> O/U {ou_side} {ou_line} ({ou_odds})"
            f" &nbsp;·&nbsp; Model projects {ou_pred} total"
            f" &nbsp;·&nbsp; Edge +{ou_edge}%</div>"
        )

    return (
        f'<div style="margin-bottom:16px;border:1px solid {BORDER_COLOR};'
        f'border-radius:8px;overflow:hidden;">'
        f'<div style="background:{color};color:white;padding:6px 14px;'
        f'font-size:11px;font-weight:700;letter-spacing:1px;">'
        f"{label} &nbsp;·&nbsp; +{edge}%"
        "</div>"
        f'<div style="padding:14px 16px;background:white;">'
        + game_line
        + f'<div style="font-size:24px;font-weight:800;color:{color};margin-bottom:8px;">'
        f'{team} &nbsp;<span style="font-size:16px;font-weight:600;color:{TEXT_COLOR};">'
        f"{odds}</span></div>"
        f'<div style="display:flex;gap:20px;font-size:13px;color:{MUTED_COLOR};margin-bottom:10px;">'
        f'<span>Edge: <strong style="color:{TEXT_COLOR};">+{edge}%</strong></span>'
        f'<span>EV: <strong style="color:{TEXT_COLOR};">+{ev}%</strong></span>'
        f'<span>Kelly: <strong style="color:{TEXT_COLOR};">{kelly}%</strong></span>'
        "</div>"
        + just_block + inj_block + ou_block
        + "</div></div>"
    )


def _ou_only_card(g: dict) -> str:
    t     = g.get("totals", {})
    oub   = t.get("best_ou_bet", {})
    side  = oub.get("side", "").upper()
    line  = t.get("market_line", "?")
    pred  = t.get("predicted_total", "?")
    edge  = round((oub.get("edge") or 0) * 100, 1)
    odds  = format_odds(oub.get("odds"))
    away  = g.get("away_team", "")
    home  = g.get("home_team", "")
    gtime = time_et(g.get("commence_time", ""))
    time_note = f" ({gtime})" if gtime else ""

    return (
        f'<div style="margin-bottom:10px;padding:12px 14px;border:1px solid {BORDER_COLOR};'
        f'border-radius:8px;background:white;">'
        f'<div style="font-weight:700;font-size:14px;">'
        f"{away} @ {home}"
        f'<span style="font-size:12px;font-weight:400;color:{MUTED_COLOR};">{time_note}</span>'
        f"</div>"
        f'<div style="font-size:13px;margin-top:6px;color:{MUTED_COLOR};">'
        f'<strong style="color:{BLUE_COLOR};">O/U {side} {line}</strong> ({odds})'
        f" &nbsp;·&nbsp; Model: {pred} total &nbsp;·&nbsp; Edge: +{edge}%"
        f"</div></div>"
    )


def build_html(data: dict) -> str:
    target_date = data.get("date", date.today().isoformat())
    games       = data.get("games", [])
    nfl         = [g for g in games if g.get("sport") == "americanfootball_nfl"]
    gen_at      = data.get("generated_at", "")[:16].replace("T", " ") + " UTC"

    dt      = datetime.strptime(target_date, "%Y-%m-%d")
    day_str = dt.strftime("%A, %B %-d, %Y")

    strong   = [g for g in nfl if g.get("best_bet") and g["best_bet"].get("strength") == "strong"]
    moderate = [g for g in nfl if g.get("best_bet") and g["best_bet"].get("strength") == "moderate"]
    all_picks = strong + moderate
    ou_only   = [g for g in nfl if not g.get("best_bet") and (g.get("totals") or {}).get("best_ou_bet")]

    if not all_picks:
        picks_html = (
            f'<div style="padding:24px;text-align:center;color:{MUTED_COLOR};'
            f'background:{BG_COLOR};border-radius:8px;">'
            f"No moneyline edges found today — market lines are efficient. "
            f"Skip or use extreme caution.</div>"
        )
    else:
        parts = []
        if strong:
            parts.append(
                f'<h3 style="color:{STRONG_COLOR};margin:20px 0 10px;font-size:14px;'
                f'text-transform:uppercase;letter-spacing:1px;">🔥 Strong Edges ({len(strong)})</h3>'
            )
            parts.extend(_bet_card(g) for g in strong)
        if moderate:
            parts.append(
                f'<h3 style="color:{MODERATE_COLOR};margin:20px 0 10px;font-size:14px;'
                f'text-transform:uppercase;letter-spacing:1px;">📊 Moderate Edges ({len(moderate)})</h3>'
            )
            parts.extend(_bet_card(g) for g in moderate)
        picks_html = "\n".join(parts)

    ou_html = ""
    if ou_only:
        ou_cards = "\n".join(_ou_only_card(g) for g in ou_only)
        ou_html = (
            f'<h3 style="color:{BLUE_COLOR};margin:24px 0 10px;font-size:14px;'
            f'text-transform:uppercase;letter-spacing:1px;">⬆️ Over/Under Plays ({len(ou_only)})</h3>'
            + ou_cards
        )

    sorted_nfl  = sorted(nfl, key=lambda x: x.get("commence_time", ""))
    rows        = "\n".join(_game_row(g) for g in sorted_nfl)
    total_picks = len(all_picks) + len(ou_only)

    games_table = (
        f'<h3 style="margin:0 0 12px;font-size:12px;color:{MUTED_COLOR};'
        f'text-transform:uppercase;letter-spacing:1px;">All NFL Games</h3>'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<tr style="color:{MUTED_COLOR};border-bottom:2px solid {BORDER_COLOR};">'
        f'<th align="left" style="padding:6px 4px;">Game</th>'
        f'<th align="right" style="padding:6px 4px;">Home%</th>'
        f'<th align="right" style="padding:6px 4px;">Away%</th>'
        f'<th align="right" style="padding:6px 4px;">Edge</th>'
        f"</tr>{rows}</table>"
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:{BG_COLOR};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
  <div style="max-width:600px;margin:0 auto;padding:16px;">

    <div style="background:linear-gradient(135deg,#1a365d 0%,#2d6a4f 100%);color:white;padding:24px 20px;border-radius:12px 12px 0 0;">
      <div style="font-size:11px;letter-spacing:2px;opacity:0.65;text-transform:uppercase;margin-bottom:4px;">Sports Edge Model</div>
      <div style="font-size:28px;font-weight:800;margin-bottom:4px;">🏈 NFL Edge Report</div>
      <div style="font-size:15px;opacity:0.85;margin-bottom:16px;">{day_str}</div>
      <div style="font-size:13px;opacity:0.8;">{len(nfl)} games analyzed &nbsp;·&nbsp; {total_picks} pick{"s" if total_picks != 1 else ""} &nbsp;·&nbsp; Generated {gen_at}</div>
    </div>

    <div style="background:white;padding:20px;border:1px solid {BORDER_COLOR};border-top:none;">
      {picks_html}
      {ou_html}
    </div>

    <div style="background:white;padding:20px;border:1px solid {BORDER_COLOR};border-top:none;border-radius:0 0 12px 12px;">
      {games_table}
    </div>

    <div style="text-align:center;padding:16px;font-size:11px;color:{MUTED_COLOR};">
      <a href="https://pdaly42.github.io/sports-edge/" style="color:{BLUE_COLOR};text-decoration:none;">View Full Dashboard</a>
      &nbsp;·&nbsp; Strong ≥8% edge &nbsp;·&nbsp; Moderate ≥3% &nbsp;·&nbsp; Max 3 picks/day
    </div>

  </div>
</body>
</html>"""


def send_email(html: str, subject: str) -> None:
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not app_password:
        print("GMAIL_APP_PASSWORD not set — skipping email send")
        print(f"Subject: {subject}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = FROM_EMAIL
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(FROM_EMAIL, app_password)
        server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
    print(f"Email sent → {TO_EMAIL}: {subject}")


def main() -> None:
    target_date = date.today().isoformat()
    json_path   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(f"predictions_{target_date}.json")

    if not json_path.exists():
        print(f"No predictions file: {json_path} — skipping email")
        return

    data  = json.loads(json_path.read_text())
    games = data.get("games", [])
    nfl   = [g for g in games if g.get("sport") == "americanfootball_nfl"]

    if not nfl:
        print("No NFL games in predictions — skipping email")
        return

    picks    = [g for g in nfl if g.get("best_bet")]
    ou_only  = [g for g in nfl if not g.get("best_bet") and g.get("totals", {}).get("best_ou_bet")]
    n_total  = len(picks) + len(ou_only)
    n_strong = sum(1 for g in picks if g["best_bet"].get("strength") == "strong")
    n_mod    = sum(1 for g in picks if g["best_bet"].get("strength") == "moderate")

    dt      = datetime.strptime(target_date, "%Y-%m-%d")
    day_str = dt.strftime("%a %b %-d")

    if n_total > 0:
        parts = []
        if n_strong:
            parts.append(f"{n_strong} strong")
        if n_mod:
            parts.append(f"{n_mod} moderate")
        if ou_only:
            parts.append(f"{len(ou_only)} O/U")
        subject = f"🏈 NFL Picks — {day_str} ({', '.join(parts)})"
    else:
        subject = f"🏈 NFL Edge Report — {day_str} (no picks today)"

    html = build_html(data)
    send_email(html, subject)


if __name__ == "__main__":
    main()
