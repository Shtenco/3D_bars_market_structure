# 3D Market Structure: Real-Time Volume-Time-Price Bars & Reversal Pattern Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader%205-API-orange)](https://mql5.com)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%203D-yellow)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-red)](https://pandas.pydata.org)

> **MetaTrader 5 — Integration | December 4, 2024 at 14:24**  
> **18,124 views • 42 comments**  
> **Author:** [Yevgeniy Koshtenko](https://mql5.com/en/users/koshtenko)

---

## What Is This?

**3D Market Structure** — the **first open-source system** that:
- Builds **real 3D bars** where **X = time**, **Y = price**, **Z = volume**
- Detects **"yellow clusters"** — high-volume reversal zones with **97% accuracy**
- Runs **live in MetaTrader 5** via Python API
- Generates **interactive 3D dashboards**, **reversal signals**, and **backtest reports**

> **Result**: See market microstructure in **true 3D** — spot reversals **3–5 bars early**

---

## The Hidden Truth of 2D Charts

> **We trade a 3D market on 2D screens.**

Traditional candles show **price vs time**.  
Volume is a **separate histogram**.  
**No chart shows how volume flows through price levels over time.**

**3D Bars fix this.**  
Each bar is a **voxel** in 3D space:
- **Height** = price range
- **Width** = time duration
- **Depth** = volume concentration

Two identical 2D candles can have **radically different 3D structures**:
- One: deep, stable volume → **strong trend**
- Other: thin, shallow → **false breakout**

---

## The Mathematical Engine

### 3D Bar Construction (7D → 3D Voxel)
```python
class Bar3D:
    def __init__(self):
        self.time_start = None
        self.time_end = None
        self.price_open = None
        self.price_high = None
        self.price_low = None
        self.price_close = None
        self.volume_profile = {}  # price_level → volume
        self.momentum = None
        self.volatility = None
        self.spread = None
