# Pattern Reference

This document describes every candlestick and chart pattern implemented in this repository, drawn from *Japanese Candlestick Charting Techniques* by Steve Nison. For each pattern you will find:

- The **signal direction** (bullish / bearish)
- The **entry conditions** (what the candle structure must satisfy)
- **Confirmation signals** Nison recommends before acting (these are *not* coded; they are for the trader's judgment)
- All **quantitative hyperparameters** introduced in this implementation that do not appear as specific numbers in the original book, along with their default values and meaning

---

## Shared Candle Geometry

All patterns use the following definitions:

| Term | Definition |
|---|---|
| **Body** | `close − open` (positive = white/bullish, negative = black/bearish) |
| **Body size** | `|close − open|` |
| **Body top** | `max(open, close)` |
| **Body bottom** | `min(open, close)` |
| **Upper shadow** | `high − body_top` |
| **Lower shadow** | `body_bottom − low` |
| **Range** | `high − low` |

**Trend filter (single-bar patterns):** `close[t] > close[t − trend_lookback]` for uptrend; `<` for downtrend.
**Trend filter (multi-bar patterns):** measured at the *first candle* of the pattern (`close[t−N]`) to avoid the pattern's own bars contaminating the reading.

---

## Module: `patterns/reversal.py` — Nison Chapters 3–5

### Hammer

**Signal:** bullish
**Pattern (1 candle):**
- Appears in a downtrend
- Body at the upper end of the range (small upper shadow)
- Long lower shadow

**Confirmation:** A gap up or strong white candle on the following session.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to define the prevailing trend (`close[t] < close[t−k]`) |
| `shadow_multiplier` | 2.0 | Lower shadow must be ≥ this multiple of the body size |
| `upper_shadow_ratio` | 0.1 | Upper shadow must be ≤ this fraction of the total range |

---

### Hanging Man

**Signal:** bearish
**Pattern (1 candle):** Identical shape to the hammer but appearing after an uptrend.

**Confirmation:** A gap down or long black candle on the following session. Nison stresses this pattern needs confirmation more than the hammer does.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Same as hammer |
| `shadow_multiplier` | 2.0 | Same as hammer |
| `upper_shadow_ratio` | 0.1 | Same as hammer |

---

### Engulfing Pattern

**Signal:** bullish or bearish
**Pattern (2 candles):**
- **Bullish:** large white body fully engulfs the prior black body, after a downtrend
- **Bearish:** large black body fully engulfs the prior white body, after an uptrend
- The current body must extend beyond both the top and bottom of the prior body

**Confirmation:** A gap in the signal direction, or a strong follow-through candle.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |

---

### Dark Cloud Cover

**Signal:** bearish
**Pattern (2 candles):**
- Prior candle is a long white candle in an uptrend
- Current candle opens *above* the prior close
- Current candle closes *below* the midpoint of the prior body (and above the prior open)

**Confirmation:** A gap down or additional black candle; the deeper the close into the prior body the stronger the signal.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `penetration` | 0.5 | Minimum fraction of the prior body that the close must penetrate (0.5 = midpoint) |

---

### Piercing Pattern

**Signal:** bullish
**Pattern (2 candles):** Mirror of dark cloud cover — long black candle followed by a white candle that opens below the prior close and closes above the midpoint of the prior body.

**Confirmation:** Gap up or strong white candle the following day.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Same as dark cloud cover |
| `penetration` | 0.5 | Same as dark cloud cover |

---

## Module: `patterns/pivots.py` — Grimes Framework

### Pivot Highs and Pivot Lows (Orders 1–3)

Not a candlestick signal — used as structural building blocks for chart-pattern detection.

**1st-order pivot high:** `high[t] > high[t−1]` and `high[t] > high[t+1]`
**2nd-order pivot high:** a 1st-order pivot high that is higher than the previous and next 1st-order pivot highs
**3rd-order pivot high:** same rule applied to 2nd-order pivots
*(Pivot lows mirror the above using lows)*

**No-lookahead rule:** A pivot at bar `t` cannot be confirmed until bar `t+1` exists. The last detected pivot in any series is always masked as unconfirmed.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `order` | 1 | Pivot order (1, 2, or 3) |

---

## Module: `patterns/stars.py` — Nison Chapters 6 and 9

### Inverted Hammer

**Signal:** bullish
**Pattern (1 candle):**
- Appears in a downtrend
- Small body at the lower end of the range
- Long upper shadow (≥ 2× body)
- Little or no lower shadow

**Confirmation:** Essential — a gap up or white candle on the next session. Without confirmation this is only potential.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `shadow_multiplier` | 2.0 | Upper shadow must be ≥ this multiple of the body size |
| `lower_shadow_ratio` | 0.1 | Lower shadow must be ≤ this fraction of total range |

---

### Shooting Star

**Signal:** bearish
**Pattern (1 candle):** Identical shape to the inverted hammer but appearing in an uptrend.

**Confirmation:** A gap down or long black candle on the following session.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Same as inverted hammer |
| `shadow_multiplier` | 2.0 | Same as inverted hammer |
| `lower_shadow_ratio` | 0.1 | Same as inverted hammer |

---

### Morning Star

**Signal:** bullish
**Pattern (3 candles):**
1. Long black candle in a downtrend
2. Small-bodied star that gaps below the first candle's body
3. White candle that closes above the midpoint of the first candle's body

**Confirmation:** The third candle itself is strong confirmation; additional follow-through (gap up, high close) adds confidence.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−2 (first candle) |
| `penetration` | 0.5 | Third candle must close above this fraction from the bottom of the first body |
| `require_gap` | True | Star body must gap entirely below the first body; set False for gapless markets |
| `small_body_ratio` | 0.3 | Star body must be ≤ this fraction of its own range |
| `long_body_ratio` | 0.6 | First and third candles must have body ≥ this fraction of their own range |

---

### Evening Star

**Signal:** bearish
**Pattern (3 candles):** Mirror of the morning star — long white, small star (gapping up), long black penetrating the midpoint of the first body.

**Confirmation:** Gap down or continued selling on subsequent sessions.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−2 |
| `penetration` | 0.5 | Third candle must close below this fraction from the top of the first body |
| `require_gap` | True | Star body must gap entirely above the first body |
| `small_body_ratio` | 0.3 | Star body size threshold |
| `long_body_ratio` | 0.6 | First and third candle body threshold |

---

### Morning Doji Star

**Signal:** bullish (stronger than morning star)
**Pattern (3 candles):** Same as morning star but the middle candle must be a doji (open ≈ close).

**Confirmation:** Same as morning star; the doji itself already implies indecision.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−2 |
| `penetration` | 0.5 | Third candle penetration threshold |
| `require_gap` | True | Gap requirement for star |
| `long_body_ratio` | 0.6 | First and third candle body threshold |
| `doji_threshold` | 0.1 | Doji: body ≤ this fraction of range |

---

### Evening Doji Star

**Signal:** bearish (stronger than evening star)
**Pattern (3 candles):** Mirror of morning doji star.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−2 |
| `penetration` | 0.5 | Third candle penetration threshold |
| `require_gap` | True | Gap requirement for star |
| `long_body_ratio` | 0.6 | First and third candle body threshold |
| `doji_threshold` | 0.1 | Doji body threshold |

---

## Module: `patterns/doji.py` — Nison Chapters 7–8

### Doji at Top

**Signal:** bearish
**Pattern (1 candle):** A doji (open ≈ close) appearing in an uptrend. Warns of potential exhaustion.

**Confirmation:** Required — a gap down, a black candle, or a close below the doji's low on the next session. A doji alone is only a warning, not a reversal.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `doji_threshold` | 0.1 | Body must be ≤ this fraction of the total range |

---

### Doji at Bottom

**Signal:** bullish
**Pattern (1 candle):** A doji appearing in a downtrend. Warns that selling pressure may be waning.

**Confirmation:** Required — a gap up or white candle on the following session.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `doji_threshold` | 0.1 | Same as doji at top |

---

### Long-Legged Doji

**Signal:** bearish (in uptrend) or bullish (in downtrend)
**Pattern (1 candle):**
- Open ≈ close (doji)
- Long upper shadow
- Long lower shadow
- Represents extreme indecision — market went far in both directions and closed unchanged

**Confirmation:** Direction and strength of the following candle.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `doji_threshold` | 0.1 | Body must be ≤ this fraction of the range |
| `shadow_ratio` | 0.3 | Both upper *and* lower shadow must be ≥ this fraction of the range |

---

### Rickshaw Man

**Signal:** bearish (in uptrend) or bullish (in downtrend)
**Pattern (1 candle):** A long-legged doji where the tiny body is also near the *midpoint* of the range. A more specific and theoretically stronger form of the long-legged doji.

**Confirmation:** Same as long-legged doji.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `doji_threshold` | 0.1 | Doji body threshold |
| `shadow_ratio` | 0.3 | Minimum shadow fraction (both sides) |
| `center_tolerance` | 0.25 | Body midpoint must be within this fraction of the range from the range midpoint |

---

### Gravestone Doji

**Signal:** bearish
**Pattern (1 candle):**
- Open ≈ close at or near the session *low*
- Long upper shadow (price tried to rally but failed completely)
- Negligible lower shadow
- Most significant at market tops in an uptrend

**Confirmation:** A weak or black candle on the following session; a gap down strongly confirms.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `doji_threshold` | 0.1 | Body must be ≤ this fraction of the range |
| `lower_shadow_ratio` | 0.05 | Lower shadow must be ≤ this fraction of the range |
| `upper_shadow_ratio` | 0.5 | Upper shadow must be ≥ this fraction of the range |

---

### Tri-Star

**Signal:** bullish (after downtrend) or bearish (after uptrend)
**Pattern (3 candles):** Three consecutive doji candles. Extremely rare; considered a very strong reversal signal when it occurs.

**Confirmation:** Follow-through on the next session.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−2 (first doji) |
| `doji_threshold` | 0.1 | All three candles must qualify as doji |

---

## Module: `patterns/more_reversals.py` — Nison Chapters 10–12

### Harami

**Signal:** bullish or bearish
**Pattern (2 candles):**
- **Bullish:** large black candle → small body contained within the first body, in a downtrend
- **Bearish:** large white candle → small body contained within the first body, in an uptrend
- Containment is strict (body top and bottom both inside the prior body)

**Confirmation:** A candle in the reversal direction on the following session.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `doji_threshold` | 0.1 | Used only when `require_doji=True` |

---

### Harami Cross

**Signal:** bullish or bearish (stronger than harami)
**Pattern (2 candles):** Same as harami but the second candle must be a doji.

**Confirmation:** Same as harami; the doji already signals indecision.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `doji_threshold` | 0.1 | Second candle: body ≤ this fraction of range |

---

### Tweezers Top

**Signal:** bearish
**Pattern (2 candles):** Two consecutive bars with matching highs after an uptrend. The market tested the same resistance twice and failed.

**Confirmation:** A black candle or gap down on subsequent sessions; significance increases when combined with other bearish signals.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `tolerance` | 0.001 | Highs match within this fraction of the prior bar's high (0.1%) |

---

### Tweezers Bottom

**Signal:** bullish
**Pattern (2 candles):** Two consecutive bars with matching lows after a downtrend.

**Confirmation:** A white candle or gap up on subsequent sessions.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `tolerance` | 0.001 | Lows match within this fraction of the prior bar's low |

---

### Belt-Hold Lines

**Signal:** bullish or bearish
**Pattern (1 candle):**
- **Bullish (white opening shaven bottom):** white candle whose open is at or near the session low — no lower shadow — with a long body, in a downtrend
- **Bearish (black opening shaven head):** black candle whose open is at or near the session high — no upper shadow — with a long body, in an uptrend

**Confirmation:** Follow-through in the signal direction. Nison notes the longer the body and the more rarely the price level has appeared, the more significant the pattern.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `shadow_ratio` | 0.05 | The shadow at the open end must be ≤ this fraction of the range |
| `long_body_ratio` | 0.6 | Body must be ≥ this fraction of the range |

---

### Upside-Gap Two Crows

**Signal:** bearish
**Pattern (3 candles):**
1. Long white candle in an uptrend
2. Black candle that gaps up above the first body and stays above it
3. Larger black candle that opens above the second candle's open and closes into the first candle's body (below its close, above its open)

**Confirmation:** Continued selling; the full close into the first body already implies strong bearish pressure.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−2 |
| `long_body_ratio` | 0.6 | First candle body threshold |

---

### Three Black Crows

**Signal:** bearish
**Pattern (3 candles):**
- Three consecutive long black candles, each opening within the prior candle's body and closing near its low

**Confirmation:** The pattern itself is strong; watch for an oversold bounce that would invalidate.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−2 |
| `long_body_ratio` | 0.6 | All three candles must have body ≥ this fraction of their range |
| `shadow_ratio` | 0.1 | Lower shadow of each candle must be ≤ this fraction of its range |

---

### Counterattack Lines

**Signal:** bullish or bearish
**Pattern (2 candles):**
- **Bullish:** black candle → white candle that opens lower (gap down) but closes at *the same price* as the prior close, in a downtrend
- **Bearish:** white candle → black candle that opens higher (gap up) but closes at the same price as the prior close, in an uptrend
- Weaker than piercing pattern / dark cloud cover because there is no penetration into the prior body

**Confirmation:** Follow-through strongly recommended given the pattern's weaker nature.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `close_tolerance` | 0.001 | Closes match within this fraction of the prior close (0.1%) |

---

### Three Mountains

**Signal:** bearish
**Pattern (multi-bar):** Three confirmed 1st-order pivot highs of approximately equal height — a triple top. Signal fires on the bar that *confirms* the third pivot (i.e., the bar after the third pivot high).

**Confirmation:** A break below the intervening lows (the "neckline") with follow-through.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `lookback` | 30 | Maximum bar span from first to third pivot high |
| `tolerance` | 0.03 | Max relative spread of the three peak highs (3%) |

---

### Three Rivers

**Signal:** bullish
**Pattern (multi-bar):** Mirror of three mountains — three confirmed pivot lows of approximately equal value.

**Confirmation:** A break above the intervening highs with follow-through.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `lookback` | 30 | Maximum bar span from first to third pivot low |
| `tolerance` | 0.03 | Max relative spread of the three trough lows (3%) |

---

### Dumpling Top

**Signal:** bearish
**Pattern (multi-bar):**
- A window of bars with small bodies forming a rounded, dome-shaped arc in the close prices
- Confirmed by a downside gap on the current bar (open < prior close)

**Confirmation:** The gap itself is part of the confirmation; continued selling after the gap.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `window` | 10 | Number of bars in the arc window (not including the gap bar) |
| `small_body_ratio` | 0.4 | All window candles must have body ≤ this fraction of their range |

Arc shape criterion (implementation-defined): the highest close must fall in the middle third of the window, and both the first and last thirds must average below the middle third.

---

### Fry Pan Bottom

**Signal:** bullish
**Pattern (multi-bar):** Mirror of dumpling top — small-bodied candles form a bowl (U) shape in closes, confirmed by an upside gap.

**Confirmation:** The gap itself plus continued buying.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `window` | 10 | Number of bars in the arc window |
| `small_body_ratio` | 0.4 | All window candles body threshold |

---

### Tower Top

**Signal:** bearish
**Pattern (multi-bar):** Across a rolling window the price structure shows three phases:
1. First third: at least one long white candle (the "tower" rising)
2. Middle third: all small-bodied candles (consolidation / topping process)
3. Last third: at least one long black candle (the "tower" falling)

**Confirmation:** Continued selling after the long black candle phase.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `window` | 10 | Total rolling window length |
| `long_body_ratio` | 0.6 | Body threshold for "long" candles |
| `small_body_ratio` | 0.3 | Body threshold for "small" candles in the middle phase |

---

### Tower Bottom

**Signal:** bullish
**Pattern (multi-bar):** Mirror of tower top — long black → small-bodied → long white.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `window` | 10 | Total rolling window length |
| `long_body_ratio` | 0.6 | Long candle body threshold |
| `small_body_ratio` | 0.3 | Small candle body threshold |

---

## Module: `patterns/continuation.py` — Nison Chapters 13–14

### Window Up (Rising Window / Upside Gap)

**Signal:** bullish continuation
**Pattern (2 bars):** Current bar's low is strictly above the prior bar's high — an upside gap. Nison considers an unfilled window as ongoing support.

**Confirmation:** The window itself is the signal. A subsequent pullback that *fails to fill* the window reinforces the bullish bias.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 0 | If > 0, requires a prior uptrend at t−1 before the gap; 0 = no trend filter (gap is self-explanatory) |

---

### Window Down (Falling Window / Downside Gap)

**Signal:** bearish continuation
**Pattern (2 bars):** Current bar's high is strictly below the prior bar's low.

**Confirmation:** Same logic — an unfilled window acts as ongoing resistance.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 0 | Same as window up |

---

### Rising Three Methods

**Signal:** bullish continuation
**Pattern (5 candles):**
1. Long white candle in an uptrend
2–4. Three small candles (traditionally black) that remain within the first candle's high–low range — a consolidation
5. Long white candle that closes above the first candle's close

**Confirmation:** The fifth candle's close above the first candle's close is the confirmation within the pattern itself.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−4 (first candle) |
| `long_body_ratio` | 0.6 | First and fifth candles body threshold |
| `small_body_ratio` | 0.3 | Middle three candles body threshold |

---

### Falling Three Methods

**Signal:** bearish continuation
**Pattern (5 candles):** Mirror of rising three methods — long black, three small candles within the first range, long black closing below the first close.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−4 |
| `long_body_ratio` | 0.6 | First and fifth candles body threshold |
| `small_body_ratio` | 0.3 | Middle three candles body threshold |

---

### Three White Soldiers

**Signal:** bullish (reversal or continuation depending on context)
**Pattern (3 candles):**
- Three consecutive long white candles after a downtrend
- Each opens within the prior candle's body
- Each closes near its high (small upper shadow)

**Confirmation:** Nison cautions that after a long advance, this pattern can lead to an overbought condition and may be followed by a correction.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Trend measured at bar t−2 (first candle) |
| `long_body_ratio` | 0.6 | All three candles body threshold |
| `shadow_ratio` | 0.1 | Upper shadow of each candle must be ≤ this fraction of its range |

---

### Separating Lines

**Signal:** bullish or bearish continuation
**Pattern (2 candles):**
- Both candles open at approximately the same price
- **Bullish:** black candle (counter-trend pullback) → white candle opening at the same price, in an uptrend
- **Bearish:** white candle (counter-trend bounce) → black candle opening at the same price, in a downtrend
- The market retraced but failed to hold, resuming the prevailing trend

**Confirmation:** Continued movement in the trend direction after the second candle.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `trend_lookback` | 5 | Bars used to identify the prevailing trend |
| `open_tolerance` | 0.001 | Opens must match within this fraction of the prior open (0.1%) |

---

## Notes on Confirmation

Nison consistently emphasizes that **candlestick signals should be confirmed** before acting. Common confirmation signals that are *not* coded in this implementation include:

- **Gap in the signal direction** on the following session
- **Volume confirmation** — a reversal bar on higher-than-average volume is more meaningful
- **Western technical support/resistance** — a pattern at a key moving average, trend line, or prior pivot carries more weight
- **Subsequent candle color** — a white candle after a bullish signal (or black after bearish) confirms follow-through
- **Failure to fill a window** — an unfilled gap reinforces the direction of the gap

These confirmations are left to the backtesting layer or to human judgment; the pattern functions here only detect the candle structure itself.
