#!/usr/bin/env python3
"""
================================================================================
 LTC SMART MONEY ANALYST — Asistente de análisis para LTC/USDT (Bybit)
================================================================================

Este script NO ejecuta órdenes ni gestiona fondos. Es una herramienta de
análisis técnico basada en "Smart Money Concepts" (SMC) que:

  1. Descarga velas (klines) de LTC/USDT desde la API pública de Bybit.
  2. Detecta estructura de mercado: swing highs/lows, BOS (Break of
     Structure) y CHoCH (Change of Character).
  3. Detecta Order Blocks (OB), Fair Value Gaps (FVG) y zonas de liquidez
     (equal highs / equal lows).
  4. Genera un sesgo (alcista / bajista / neutral) con zonas de interés y
     niveles de invalidación.
  5. Presenta el resultado narrado por una "persona" — un analista senior
     configurable — de forma determinista (plantilla) o, si configuras tu
     propia API key de Anthropic, usando el modelo para redactar el
     análisis en lenguaje natural a partir de los datos ya calculados.

------------------------------------------------------------------------------
AVISO IMPORTANTE / DISCLAIMER
------------------------------------------------------------------------------
- Esta herramienta es EDUCATIVA e INFORMATIVA. No es asesoría financiera.
- No ejecuta órdenes ni tiene acceso a tus fondos.
- El análisis técnico (incluido SMC) NO garantiza resultados. Los mercados
  de criptomonedas son altamente volátiles y puedes perder capital.
- Antes de operar con dinero real, valida cualquier estrategia con
  backtesting y/o paper trading, y define tu propio manejo de riesgo.
------------------------------------------------------------------------------

Requisitos:
    pip install requests pandas numpy

Uso básico:
    python ltc_smc_analyst.py --interval 15 --limit 300

Uso con narrativa generada por IA (opcional, requiere tu propia API key):
    export ANTHROPIC_API_KEY="tu-api-key"
    python ltc_smc_analyst.py --interval 60 --ai-narrative

Modo vigilancia (repite el análisis cada N minutos):
    python ltc_smc_analyst.py --interval 15 --watch 5

Envío automático por Telegram (para despliegue 24/7 en Railway/Render):
    export TELEGRAM_BOT_TOKEN="tu-token-de-botfather"
    export TELEGRAM_CHAT_ID="tu-chat-id"
    python ltc_smc_analyst.py --interval 15 --watch 15 --telegram
================================================================================
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Literal

import numpy as np
import pandas as pd
import requests

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

BYBIT_BASE_URL = "https://api.bybit.com"
SYMBOL = "LTCUSDT"
CATEGORY = "spot"  # usa "spot" para mercado spot; "linear" para perpetuos USDT

# Persona del "analista experto" — puedes editar esto libremente
ANALYST_PERSONA = """
Eres Marcus Kane, analista senior de mercados con más de 30 años de
experiencia combinada entre Wall Street (trading de renta fija y FX en los
años 90-2000) y, desde 2013, análisis cuantitativo de criptomonedas.
Tu estilo es directo, profesional, sin hype ni promesas de ganancias.
Explicas el "por qué" detrás de cada zona usando Smart Money Concepts
(estructura de mercado, order blocks, fair value gaps, liquidez).
Siempre aclaras el nivel de invalidación de tu lectura y recuerdas que
esto es análisis técnico, no una recomendación de inversión personalizada.
"""

# ==============================================================================
# 1. DESCARGA DE DATOS (Bybit v5 API pública — no requiere autenticación)
# ==============================================================================


def fetch_klines(symbol: str = SYMBOL, interval: str = "15", limit: int = 300,
                  category: str = CATEGORY) -> pd.DataFrame:
    """
    Descarga velas OHLCV desde Bybit.

    interval: "1","3","5","15","30","60","120","240","360","720","D","W","M"
    limit: número de velas (máx 1000 por Bybit)
    """
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")

    rows = data["result"]["list"]
    if not rows:
        raise RuntimeError("Bybit no devolvió datos. Revisa symbol/interval.")

    # Bybit devuelve más reciente primero -> invertimos
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    df = df.iloc[::-1].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ==============================================================================
# 2. ESTRUCTURA DE MERCADO (Swing points, BOS, CHoCH)
# ==============================================================================


@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["high", "low"]


@dataclass
class StructureEvent:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["BOS_bullish", "BOS_bearish", "CHoCH_bullish", "CHoCH_bearish"]


def find_swing_points(df: pd.DataFrame, order: int = 4) -> List[SwingPoint]:
    """
    Detecta swing highs/lows tipo fractal: un high es swing si es el máximo
    de una ventana [-order, +order] alrededor suyo (equivalente para lows).
    """
    swings: List[SwingPoint] = []
    highs, lows = df["high"].values, df["low"].values

    for i in range(order, len(df) - order):
        window_h = highs[i - order: i + order + 1]
        window_l = lows[i - order: i + order + 1]

        if highs[i] == window_h.max() and highs[i] == window_h[order]:
            swings.append(SwingPoint(i, df["timestamp"][i], highs[i], "high"))
        if lows[i] == window_l.min() and lows[i] == window_l[order]:
            swings.append(SwingPoint(i, df["timestamp"][i], lows[i], "low"))

    swings.sort(key=lambda s: s.index)
    return swings


def detect_structure_events(swings: List[SwingPoint]) -> List[StructureEvent]:
    """
    Recorre la secuencia de swings y clasifica BOS / CHoCH:

    - Tendencia alcista = secuencia de Higher Highs (HH) + Higher Lows (HL).
      BOS_bullish: se rompe el último HH -> continuación alcista.
      CHoCH_bearish: se rompe el último HL -> posible cambio a bajista.

    - Tendencia bajista = secuencia de Lower Lows (LL) + Lower Highs (LH).
      BOS_bearish: se rompe el último LL -> continuación bajista.
      CHoCH_bullish: se rompe el último LH -> posible cambio a alcista.
    """
    events: List[StructureEvent] = []
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]

    trend: Optional[str] = None  # "up" | "down"
    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None

    merged = sorted(swings, key=lambda s: s.index)
    for s in merged:
        if s.kind == "high":
            if last_high is not None:
                if s.price > last_high.price:
                    if trend == "down":
                        events.append(StructureEvent(s.index, s.timestamp, s.price, "CHoCH_bullish"))
                    elif trend == "up":
                        events.append(StructureEvent(s.index, s.timestamp, s.price, "BOS_bullish"))
                    trend = "up"
            last_high = s
        else:
            if last_low is not None:
                if s.price < last_low.price:
                    if trend == "up":
                        events.append(StructureEvent(s.index, s.timestamp, s.price, "CHoCH_bearish"))
                    elif trend == "down":
                        events.append(StructureEvent(s.index, s.timestamp, s.price, "BOS_bearish"))
                    trend = "down"
            last_low = s

    return events


# ==============================================================================
# 3. ORDER BLOCKS, FAIR VALUE GAPS Y LIQUIDEZ
# ==============================================================================


@dataclass
class Zone:
    kind: str            # "order_block_bullish" | "order_block_bearish" | "fvg_bullish" | "fvg_bearish" | "liquidity_high" | "liquidity_low"
    top: float
    bottom: float
    index: int
    timestamp: pd.Timestamp
    note: str = ""


def find_order_blocks(df: pd.DataFrame, events: List[StructureEvent], lookback: int = 8) -> List[Zone]:
    """
    Order Block simplificado: la última vela opuesta antes del movimiento
    impulsivo que provocó un BOS/CHoCH.
      - BOS/CHoCH alcista -> buscamos la última vela BAJISTA antes del impulso.
      - BOS/CHoCH bajista -> buscamos la última vela ALCISTA antes del impulso.
    """
    zones: List[Zone] = []
    for ev in events:
        start = max(0, ev.index - lookback)
        segment = df.iloc[start:ev.index]
        if segment.empty:
            continue

        bullish_event = "bullish" in ev.kind
        if bullish_event:
            bearish_candles = segment[segment["close"] < segment["open"]]
            if not bearish_candles.empty:
                c = bearish_candles.iloc[-1]
                zones.append(Zone(
                    "order_block_bullish", top=c["open"], bottom=c["low"],
                    index=int(c.name), timestamp=c["timestamp"],
                    note=f"OB alcista previo a {ev.kind} en {ev.timestamp:%Y-%m-%d %H:%M}",
                ))
        else:
            bullish_candles = segment[segment["close"] > segment["open"]]
            if not bullish_candles.empty:
                c = bullish_candles.iloc[-1]
                zones.append(Zone(
                    "order_block_bearish", top=c["high"], bottom=c["open"],
                    index=int(c.name), timestamp=c["timestamp"],
                    note=f"OB bajista previo a {ev.kind} en {ev.timestamp:%Y-%m-%d %H:%M}",
                ))
    return zones


def find_fair_value_gaps(df: pd.DataFrame) -> List[Zone]:
    """
    FVG (patrón de 3 velas): hueco entre el high de la vela 1 y el low de la
    vela 3 (FVG alcista), o entre el low de la vela 1 y el high de la vela 3
    (FVG bajista), sin que la vela 2 lo rellene.
    """
    zones: List[Zone] = []
    for i in range(2, len(df)):
        c1, c3 = df.iloc[i - 2], df.iloc[i]
        if c1["high"] < c3["low"]:
            zones.append(Zone(
                "fvg_bullish", top=c3["low"], bottom=c1["high"],
                index=i, timestamp=df.iloc[i - 1]["timestamp"],
                note="Fair Value Gap alcista (posible zona de reequilibrio)",
            ))
        elif c1["low"] > c3["high"]:
            zones.append(Zone(
                "fvg_bearish", top=c1["low"], bottom=c3["high"],
                index=i, timestamp=df.iloc[i - 1]["timestamp"],
                note="Fair Value Gap bajista (posible zona de reequilibrio)",
            ))
    return zones


def find_liquidity_pools(swings: List[SwingPoint], tolerance_pct: float = 0.05) -> List[Zone]:
    """
    Detecta "equal highs" / "equal lows": swings muy cercanos entre sí en
    precio, que suelen actuar como imanes de liquidez (stops agrupados).
    """
    zones: List[Zone] = []
    for kind, pts in (("high", [s for s in swings if s.kind == "high"]),
                       ("low", [s for s in swings if s.kind == "low"])):
        pts = sorted(pts, key=lambda s: s.price)
        used = set()
        for i, a in enumerate(pts):
            if a.index in used:
                continue
            cluster = [a]
            for b in pts[i + 1:]:
                if abs(b.price - a.price) / a.price * 100 <= tolerance_pct:
                    cluster.append(b)
            if len(cluster) >= 2:
                for p in cluster:
                    used.add(p.index)
                avg_price = float(np.mean([p.price for p in cluster]))
                zones.append(Zone(
                    f"liquidity_{kind}", top=avg_price, bottom=avg_price,
                    index=cluster[-1].index, timestamp=cluster[-1].timestamp,
                    note=f"{len(cluster)} swings {kind} agrupados cerca de {avg_price:.2f}",
                ))
    return zones


# ==============================================================================
# 4. GENERACIÓN DE SEÑAL / SESGO
# ==============================================================================


@dataclass
class SignalReport:
    symbol: str
    timeframe: str
    price: float
    bias: str
    confidence: str
    last_event: Optional[StructureEvent]
    nearby_zones: List[Zone]
    invalidation: Optional[float]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def generate_signal(df: pd.DataFrame, events: List[StructureEvent], zones: List[Zone]) -> SignalReport:
    price = float(df["close"].iloc[-1])
    last_event = events[-1] if events else None

    if last_event is None:
        bias, confidence = "neutral", "baja"
    elif "bullish" in last_event.kind:
        bias = "alcista"
        confidence = "media-alta" if last_event.kind.startswith("BOS") else "media"
    else:
        bias = "bajista"
        confidence = "media-alta" if last_event.kind.startswith("BOS") else "media"

    # zonas cercanas al precio actual (dentro de un 3%)
    nearby = [
        z for z in zones
        if min(abs(z.top - price), abs(z.bottom - price)) / price <= 0.03
    ]
    nearby.sort(key=lambda z: min(abs(z.top - price), abs(z.bottom - price)))

    invalidation = None
    if last_event is not None:
        invalidation = last_event.price

    return SignalReport(
        symbol=SYMBOL,
        timeframe="",
        price=price,
        bias=bias,
        confidence=confidence,
        last_event=last_event,
        nearby_zones=nearby[:5],
        invalidation=invalidation,
    )


# ==============================================================================
# 5. NARRATIVA — plantilla determinista (por defecto) o vía API de Anthropic
# ==============================================================================


def render_template_report(report: SignalReport, interval: str) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"  MARCUS KANE — Análisis SMC | {report.symbol} | TF {interval}m")
    lines.append(f"  {report.generated_at:%Y-%m-%d %H:%M UTC}")
    lines.append("=" * 78)
    lines.append(f"Precio actual: {report.price:.4f} USDT")
    lines.append(f"Sesgo de estructura: {report.bias.upper()}  (confianza: {report.confidence})")

    if report.last_event:
        lines.append(
            f"Último evento de estructura: {report.last_event.kind} en "
            f"{report.last_event.price:.4f} ({report.last_event.timestamp:%Y-%m-%d %H:%M} UTC)"
        )
        lines.append(f"Nivel de invalidación de esta lectura: {report.invalidation:.4f}")
    else:
        lines.append("Sin eventos de estructura claros en el rango analizado.")

    lines.append("")
    lines.append("Zonas de interés cercanas al precio actual:")
    if report.nearby_zones:
        for z in report.nearby_zones:
            lines.append(f"  - [{z.kind}] {z.bottom:.4f} - {z.top:.4f}  | {z.note}")
    else:
        lines.append("  - No hay zonas SMC relevantes dentro del 3% del precio actual.")

    lines.append("")
    lines.append(
        "Lectura de Marcus: "
        + (
            f"La estructura reciente favorece un sesgo {report.bias}. "
            f"Vigila la reacción del precio en las zonas listadas arriba; "
            f"una ruptura clara del nivel de invalidación ({report.invalidation:.4f}) "
            "cuestionaría este sesgo."
            if report.last_event else
            "No hay una estructura direccional clara todavía; conviene esperar "
            "confirmación antes de operar."
        )
    )
    lines.append("")
    lines.append(
        "⚠ Esto es análisis técnico educativo, no asesoría financiera personalizada. "
        "No hay garantía de resultados. Define tu propio manejo de riesgo."
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def render_ai_report(report: SignalReport, interval: str) -> str:
    """
    Usa la API de Anthropic (tu propia API key) para redactar el análisis en
    lenguaje natural a partir de los datos YA CALCULADOS por este script.
    El modelo no inventa precios: solo narra los datos que le pasamos.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[Aviso] No se encontró ANTHROPIC_API_KEY, usando reporte por plantilla.\n")
        return render_template_report(report, interval)

    try:
        import anthropic  # pip install anthropic
    except ImportError:
        print("[Aviso] Instala el SDK con 'pip install anthropic' para usar --ai-narrative. "
              "Usando reporte por plantilla.\n")
        return render_template_report(report, interval)

    client = anthropic.Anthropic(api_key=api_key)

    data_summary = {
        "symbol": report.symbol,
        "timeframe_minutes": interval,
        "price": report.price,
        "bias": report.bias,
        "confidence": report.confidence,
        "last_structure_event": (
            {
                "type": report.last_event.kind,
                "price": report.last_event.price,
                "timestamp": str(report.last_event.timestamp),
            } if report.last_event else None
        ),
        "invalidation_level": report.invalidation,
        "nearby_zones": [
            {"kind": z.kind, "top": z.top, "bottom": z.bottom, "note": z.note}
            for z in report.nearby_zones
        ],
    }

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=ANALYST_PERSONA,
        messages=[{
            "role": "user",
            "content": (
                "Aquí están los datos de análisis SMC ya calculados (no inventes "
                "otros precios ni zonas, usa exactamente estos):\n\n"
                f"{data_summary}\n\n"
                "Redacta un reporte breve y profesional en español, con el sesgo, "
                "las zonas relevantes y el nivel de invalidación. Cierra con un "
                "recordatorio de que esto no es asesoría financiera."
            ),
        }],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


# ==============================================================================
# 6. NOTIFICACIONES POR TELEGRAM
# ==============================================================================


def send_telegram_message(text: str) -> None:
    """
    Envía `text` al chat configurado vía TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
    Telegram limita mensajes a 4096 caracteres; si el reporte es más largo,
    lo recorta con seguridad.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[Aviso] Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID. No se envió mensaje.")
        return

    if len(text) > 4000:
        text = text[:3990] + "\n...(recortado)"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        if not result.get("ok"):
            print(f"[Error Telegram] {result}")
    except Exception as exc:
        print(f"[Error enviando a Telegram] {exc}")


# ==============================================================================
# 7. CLI
# ==============================================================================


def run_once(interval: str, limit: int, use_ai: bool, use_telegram: bool) -> None:
    df = fetch_klines(interval=interval, limit=limit)
    swings = find_swing_points(df, order=4)
    events = detect_structure_events(swings)

    zones = []
    zones += find_order_blocks(df, events)
    zones += find_fair_value_gaps(df)
    zones += find_liquidity_pools(swings)

    report = generate_signal(df, events, zones)

    text = render_ai_report(report, interval) if use_ai else render_template_report(report, interval)
    print(text)

    if use_telegram:
        send_telegram_message(text)


def main():
    parser = argparse.ArgumentParser(description="Asistente de análisis SMC para LTC/USDT (Bybit)")
    parser.add_argument("--interval", default="15",
                         help="Timeframe en minutos ('1','5','15','60','240','D', etc.)")
    parser.add_argument("--limit", type=int, default=300, help="Número de velas a analizar (máx 1000)")
    parser.add_argument("--ai-narrative", action="store_true",
                         help="Usa tu ANTHROPIC_API_KEY para redactar el reporte en lenguaje natural")
    parser.add_argument("--telegram", action="store_true",
                         help="Envía el reporte a tu Telegram (requiere TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID)")
    parser.add_argument("--watch", type=int, default=0,
                         help="Repite el análisis cada N minutos (0 = ejecutar una sola vez)")
    args = parser.parse_args()

    print(
        "⚠ Herramienta educativa. No ejecuta órdenes ni gestiona fondos. "
        "No es asesoría financiera.\n"
    )

    if args.watch <= 0:
        run_once(args.interval, args.limit, args.ai_narrative, args.telegram)
        return

    # Bucle continuo pensado para correr 24/7 en un worker (Railway/Render).
    # Los errores de red puntuales NO deben tumbar el proceso: se registran
    # y se reintenta en el siguiente ciclo.
    while True:
        try:
            run_once(args.interval, args.limit, args.ai_narrative, args.telegram)
        except KeyboardInterrupt:
            print("\nDetenido por el usuario.")
            sys.exit(0)
        except Exception as exc:
            print(f"[Error en el ciclo de análisis] {exc}")

        print(f"\n(Próxima actualización en {args.watch} min)\n")
        time.sleep(args.watch * 60)


if __name__ == "__main__":
    main()
