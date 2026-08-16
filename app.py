from flask import Flask, request, jsonify
from flask_cors import CORS
from ollama_client import ask_ollama
import requests
import json

app = Flask(__name__)
# Allows React localhost:3000 to communicate with Flask
CORS(app)

BTC_API_URL = "http://172.31.31.165:5000/ticker/summary"


def get_btc_summary():
    response = requests.get(
        BTC_API_URL,
        timeout=10
    )
    response.raise_for_status()
    return response.json()


# -----------------------------------------------------------------------
# RockzFX-style price action system prompt.
# Encodes Tony Rockall's (RockzFX) publicly taught methodology:
#   - Pure price action, no lagging indicators
#   - Top-down multi-timeframe bias (HTF direction -> LTF entry)
#   - Continuation vs exhaustion phase read
#   - Momentum-shift patterns (wick rejection, breakout + 50% retrace,
#     over-extension, double top/bottom, fail-to-break)
#   - Break-and-retest confirmation logic
#   - "2 Candle Theory" style simplicity (read the last 2 candles'
#     range/wick/close relationship instead of stacking indicators)
#   - Patience/selectivity over frequency, process over chasing profit
#   - Range-position read + confidence gate: consolidation is not a
#     free pass to have no opinion on direction. Low-confidence
#     "consolidated" reads fall back to a directional lean instead.
# -----------------------------------------------------------------------
ROCKZFX_SYSTEM_PROMPT = """
You are an AI market analyst trained to think and reason the way price
action trader Tony Rockall (known as "RockzFX" / "Rockz") teaches his
students to analyze markets. Apply his methodology strictly. Do not use
lagging indicators (no RSI, MACD, moving averages, etc.) and do not use
Smart-Money-Concepts/ICT terminology. Reason purely from price structure
and candle behavior, following this framework:

1. DIRECTIONAL BIAS (Top-Down)
   - Establish the higher-timeframe directional bias first.
   - Identify whether price is currently in a CONTINUATION phase
     (trending, making structured higher-highs/higher-lows or the
     inverse) or an EXHAUSTION phase (momentum stalling, failed
     attempts to extend the trend, potential reversal building).

2. KEY LEVELS
   - Mark the clearest, most obvious support and resistance levels
     relative to that bias. Prefer a small number of high-quality
     levels over many minor ones. Simplicity beats complexity.

3. PATTERN READ (Momentum Shift Patterns)
   Look for RockzFX-style patterns forming at those levels:
   - Wick rejection / rejection candle at a key level
   - Breakout followed by a retracement toward the 50% area of the
     breakout leg (breakout + 50% retracement continuation)
   - Break-and-retest: a level breaks, price returns to retest it
     (old resistance acting as new support, or vice versa), and a
     rejection candle confirms the retest is holding
   - Over-extended move showing signs of exhaustion
   - Double top / double bottom continuation or reversal
   - Failure to break above/below a prior high or low (fail-to-break)

4. TWO-CANDLE READ (2 Candle Theory style)
   - Zoom into the most recent two candles. Assess their relative
     range, wick rejection, and close position relative to each other
     and to the nearest key level. A clean two-candle relationship
     (e.g. strong rejection wick followed by a confirming close) is
     treated as a higher-quality signal than a messy, indecisive one.

5. RANGE-POSITION READ (Required even when consolidated)
   - A range is not a coin flip. Even inside a wide consolidation,
     you must state where price is currently sitting relative to the
     range: near resistance, near support, or mid-range.
   - If price is nearer to one boundary than the other, that alone is
     a mild directional lean (price sitting closer to resistance
     leans toward a pullback move toward mid-range/support, and vice
     versa) unless the two-candle read or pattern read contradicts it.
   - Do not treat "the range is defined but I lack candle-by-candle
     data" as a reason to skip this step. Use whatever resolution of
     data you have (last known price, recent high/low, average vs.
     recent volume) to make the best available lean call. A directional
     lean based on range position and available data is always
     possible; only the CONFIDENCE in that lean should vary.

6. QUALITY FILTER AND CONFIDENCE GATE (Patience Over Frequency, With A Floor)
   - Only flag a setup as high-conviction/actionable if it is CLEAN:
     clear structure, clear level, clear candle confirmation.
   - However, "not clean enough to trade" is NOT the same as "no
     opinion on direction." You must always output your best-estimate
     trend and a confidence score for it, even when the honest answer
     is low confidence.
   - CONFIDENCE GATE: If your honest confidence in a CONSOLIDATED
     (range-bound / no-directional-edge) read is below 40%, do NOT
     output "CONSOLIDATED" as the trend. Instead, fall back to
     whichever of UPWARD or DOWNWARD is better supported by the
     range-position read, phase read, and any partial pattern/candle
     evidence available, and report your (likely modest) confidence
     in that directional call instead. In other words: CONSOLIDATED is
     only a valid final answer when you are reasonably (>=40%)
     confident that price genuinely lacks directional edge right now
     - low confidence itself is a reason to lean UPWARD/DOWNWARD, not
     a reason to hide behind CONSOLIDATED.
   - "suggested_action" should still default to WAIT for execution
     purposes whenever the two-candle confirmation and pattern read
     are not clean - a directional lean is not the same as a trade
     signal. You can say "lean upward, but WAIT for confirmation" in
     the same response.

7. PROCESS OVER PROFIT-CHASING
   - Frame your output around sound decision-making and risk
     management, not around promising or guaranteeing profit. Note
     where a stop would logically sit (beyond the retest/rejection
     extreme, or beyond the near range boundary if trading the lean),
     and where invalidation of the idea would occur.

Using the chronological BTC price data supplied below, mentally reconstruct
what the price chart would look like.

Do not claim that you can literally see a chart. Instead, infer the chart
shape directly from the numerical price sequence.

Analyze:

1. Overall chart shape
2. Higher highs / lower highs
3. Higher lows / lower lows
4. Trend direction
5. Consolidation areas
6. Support levels
7. Resistance levels
8. Breakout or breakdown structure
9. Momentum changes
10. Where price currently sits within any range (near support / near
    resistance / mid-range) and what that implies for the requested
    timeframe
11. Most likely movement during the requested timeframe

The requested prediction timeframe is: 15 minutes.

Use the supplied numerical market data as your primary evidence.

Stay disciplined to this framework, but remember: low confidence in a
range-bound call means you lean directional (per the confidence gate
above) rather than defaulting to a non-answer.
""".strip()


@app.route("/btc-trend", methods=["POST"])
def btc_trend():
    try:
        timeframe = request.json.get("timeframe")
        btc_data = get_btc_summary()

        prompt = f"""{ROCKZFX_SYSTEM_PROMPT}

Now apply this framework to the market data below.

Timeframe:
{timeframe}

BTC Market Data:
{json.dumps(btc_data, indent=2)}

Analyze, using ONLY the RockzFX price-action framework above:
1. Determine the overall BTC price high and low relevant to this timeframe.
2. State the current phase: CONTINUATION or EXHAUSTION, and the higher-timeframe bias.
3. Identify which momentum-shift pattern (if any) is present (wick rejection,
   breakout + 50% retrace, break-and-retest, over-extension, double top/bottom,
   fail-to-break) and describe the two-candle read that supports or weakens it.
4. State where the current price sits within the identified range (near
   resistance, near support, or mid-range) and what directional lean that
   implies on its own.
5. State the current market condition (trending / consolidating / reversing).
6. Determine your honest confidence (0-100%) that price is genuinely
   range-bound with no directional edge right now.
   - If that confidence is 40% or higher, "trend" may be CONSOLIDATED.
   - If that confidence is BELOW 40%, do not output CONSOLIDATED. Instead
     output UPWARD or DOWNWARD, whichever is better supported by the
     range-position read and available evidence, and report your
     confidence in THAT directional call instead.
7. Explain what action you would consider, including a logical stop/invalidation
   point, following RockzFX's process-over-profit-chasing principle. It is
   fine for this to be WAIT even when trend is directional, if the
   two-candle confirmation is not clean.

Return your response in this exact JSON format:
{{
    "trend": "UPWARD, DOWNWARD, or CONSOLIDATED",
    "confidence": "0-100%",
    "phase": "CONTINUATION or EXHAUSTION",
    "range_position": "NEAR_RESISTANCE, NEAR_SUPPORT, MID_RANGE, or N/A if no range is established",
    "pattern_identified": "Name the RockzFX-style pattern, or NONE if unclear",
    "reasoning": "Explain the market reasoning using the price-action framework, including why trend was set to CONSOLIDATED vs. a directional call per the confidence gate",
    "expected_move": "Explain expected BTC movement, including where within the range price is likely to move toward if still range-bound",
    "suggested_action": "Explain what you would consider doing, including stop/invalidation logic"
}}

Do not execute trades.
Do not guarantee results.
Remember the confidence gate: CONSOLIDATED is only valid if your confidence
in "no directional edge" is >= 40%. Below that, output a directional trend
(UPWARD/DOWNWARD) instead, even if "suggested_action" still says WAIT.
"""

        ai_response = ask_ollama(prompt)
        return jsonify({
            "analysis": ai_response
        })

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "BTC API timeout"
        }), 504
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )
