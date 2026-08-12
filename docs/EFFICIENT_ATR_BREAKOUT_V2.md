# Efficient ATR breakout v2

This campaign is the bounded beginner-to-advanced validation path for the 4h
Turtle/ATR family. It is research-only and has no order, paper-promotion or live
authority.

## Levels

1. **Beginner — frozen hypothesis.** The family is long-only spot, uses prior
   Donchian channels, a completed-close EMA trend filter, ATR(14) risk sizing and
   next-open execution. The grid contains exactly 24 pre-registered DNA values.
2. **Intermediate — cheap causal screen.** Stage-0 applies the same signal,
   sizing, exposure and cost semantics in a lightweight loop. It ranks candidates
   but is explicitly not promotion evidence.
3. **Advanced — exact development evidence.** At most three Stage-0 survivors
   enter the canonical exact portfolio engine under normal and doubled costs.
4. **Expert — one untouched test.** One winner is frozen before the final
   chronological test. A failed test rejects the campaign epoch; no alternative
   DNA may be tried on that consumed holdout.
5. **Autonomous routing.** Only a future campaign that passes every gate may be
   reviewed for SHADOW or paper integration. Live promotion remains separate and
   is never implied by this campaign.

## Current result

The development winner was the 240/90 channel, EMA(1200), three-bar rebalance
cadence and 5% rebalance buffer. On development data it returned 99.01% under
normal costs and 76.01% under doubled costs. The untouched period from
2025-03-17 through 2026-08-11 failed:

- normal: -4.40%, profit factor 0.762;
- doubled costs: -6.23%, profit factor 0.674.

The epoch is therefore rejected. It produced and submitted zero orders. The
holdout is consumed and must not be reused to select a second candidate.

The shared adaptive data loader also now recognizes the normalized Parquet
`timestamp` column. Earlier RangeIndex-to-datetime conversion could collapse
calendar metrics into 1970 nanoseconds. Reports generated before this repair are
not authoritative for calendar-based evidence. The autonomous control-plane
fallback uses the canonical OHLCV loader as well, so local daily fallback data
cannot silently inherit the same timestamp error.

## Operator commands

```powershell
python main.py lab campaign plan --name efficient-atr-breakout-v2
python main.py lab campaign run --name efficient-atr-breakout-v2 --yes
python main.py lab campaign status --name efficient-atr-breakout-v2
```

The current evidence artifact is
`output/lab/reports/efficient_atr_breakout_v2.json`. Re-running it against the
same consumed test is reproducibility only, not a new independent validation.
