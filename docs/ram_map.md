# Life Force (NES) — RAM map for RL

What the agent can read from the game's 2 KB RAM. The bundled stable-retro
integration only maps `score` + `lives`; everything else here we found via
[`tools/ram_hunt.py`](../tools/ram_hunt.py) (constant-byte and monotonic-byte
scans) and cross-checked against the public
[Data Crystal RAM map](https://datacrystal.tcrf.net/wiki/Life_Force_(NES)/RAM_map).

## Confirmed (used / usable now)

| Address | Bytes | Meaning | How confirmed |
|--------:|-------|---------|---------------|
| `0x0034` (52)   | 1  | **P1 lives** | bundled integration; scan reads 2 at Stage-1 start; Data Crystal |
| `0x07E4` (2020) | 4* | **P1 score** (BCD digits 0x7E4..0x7E6) | bundled integration (`<d4`); Data Crystal |
| `0x0350` (848)  | 1  | **P1 X position** (15..232) | scan: climbed 96→232 when holding RIGHT (hits documented max); Data Crystal |
| `0x032F` (815)  | 1  | **P1 Y position** (24..197) | scan; Data Crystal |
| `0x002F` (47)   | 1  | **auto-scroll clock** — increments ~0.25/frame regardless of input, wraps at 256 | scan (monotonic, input-independent) |

\* score is 3 meaningful BCD bytes; the bundled integration reads it as a 4-byte
little-endian value at `0x07E4`.

## Stage-transition suspects (unconfirmed — need a Stage-2 observation)

| Address | Stage-1 value | Hypothesis |
|--------:|--------------:|------------|
| `0x0023` (35) | `0` | "Demo Stage Num?" (Data Crystal) — possibly 0-indexed current stage |
| `0x0040` (64) | `0` | "Is Stage Vertical?" — **Stage 2 of Life Force is vertical, so this should flip 0→1 on clearing Stage 1** |

Both read `0` on Stage 1. Detecting "Level 1 cleared" means watching one (or
both) change. We can't reach Stage 2 by scripted play, so:

## Plan to confirm (bootstrap)

1. Build the integration with the **confirmed** signals only (score reward,
   lives-based done, optional X-position progress shaping).
2. Train. The first episode that reaches Stage 2 triggers a RAM capture of the
   transition.
3. Diff that capture against the Stage-1 baseline (`ram_dumps/stage1_baseline.npz`)
   to confirm which of `0x23` / `0x40` flips — then wire up the
   "Level 1 cleared" done/reward condition.

The agent's own progress produces the Stage-2 reference we can't get by hand.

## Reproduce

```bash
python tools/ram_hunt.py baseline --steps 600   # capture + analyze Stage 1
```
