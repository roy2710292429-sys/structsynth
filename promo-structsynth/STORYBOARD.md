# STORYBOARD

**Format:** 1080×1920 portrait, 30fps  
**Audio:** No voice. Offline SFX only; no fabricated BGM.  
**Style basis:** DESIGN.md  
**Anchor:** "Same reasoning skeleton. New surface. Same depth."

## Asset Audit

| Asset | Type | Beat | Role |
|---|---|---:|---|
| assets/sfx/whoosh-cinematic.mp3 | SFX | 1, 3 | Formula-world transition |
| assets/sfx/impact-bass-1.mp3 | SFX | 1, 4, 6 | Emphasis hit |
| assets/sfx/riser.mp3 | SFX | 4 → 5 | Build into product reveal |
| assets/sfx/glitch-1.mp3 | SFX | 2, 3 | Token morph / matrix activation |
| assets/brand/mark.svg | SVG | 1, 6 | StructSynth mark |

## BEAT 1 — HOOK / 0.0–4.0

**Concept:** The first frame is already the thesis. A dark green equation world cracks open: `2^x` flips into `3^x`, while a spectral graph survives the change. The viewer understands "surface changes, structure doesn't" before reading a paragraph.

**On screen:** `换掉题面。` slams in, then `保留思维。` lands in mint. `2^x` becomes `3^x`; grid lines snap around both.

**Direction:** High-energy editorial mathematics. Paper card slices emerge from dark field. Use kinetic type, SVG path drawing, and seeded node twinkle. Camera punches in 1→1.04. Exit by formula card scaling past the frame.

## BEAT 2 — FUNCTION TRANSFORM / 4.0–14.0

**Concept:** A real exam-style function problem is treated like an architecture diagram. The original formula `f(x)=2^x` migrates to `f(x)=3^x`, while `D(x₀)`, odd symmetry, subset proof and monotonicity remain lit.

**On screen:** Left/upper formula paper enters with domain, definition of D and three subquestions compressed into readable blocks. Then base `2` flips to `3`; a coral scanner marks unchanged structures. A generated variant panel appears showing the stable skeleton.

**Direction:** Paper world on `#f2f0e9`. Struct green scanner lines. Use masked reveals, per-character morph, and SVG annotations. Keep formulas large; no tiny full transcript. Exit with a vertical spectral wipe.

## BEAT 3 — STRUCTURE ENGINE / 14.0–23.0

**Concept:** The page's engine becomes a living graph. Segments are nodes; semantic links appear mint, role-structure links appear gold; the graph contracts into eigenvalues and slot rails.

**On screen:** Chinese labels `切分 / 角色 / 邻域 / 谱聚类` snap along the left edge. Nodes form clusters; `M`, `N`, `A=2sym(M)-sym(N)` draw in mono. Candidate units become ordered semantic slots.

**Direction:** Dark void green. Continuous graph camera move. Use Canvas/GSAP node network, SVG strokes, and number readouts. Not a generic neural network: no glowing brains, no robot imagery. Transition by all nodes collapsing into a single coral point.

## BEAT 4 — CMO PEAK / 23.0–32.0

**Concept:** The coral point explodes into a dense olympiad inequality. The product isn't just making simpler lookalikes: it can represent a high-depth problem as operators, constraints and a global proof bottleneck.

**On screen:** `e₂² ≤ λ e₂e₃ + (3/n)V` appears large. Terms light individually with labels `e₂ / e₃ / V / λ`. A depth gauge rises toward 5. Text: `真正难的，是迁移整条推理链。`

**Direction:** Mathematical blackboard turned into a pressure chamber. Formula sheets collide, but never become illegible. Use line-by-line formula reveal, depth gauge, fine graph traces. No claim of a finished proof. Transition with a hard flash into paper UI.

## BEAT 5 — PRODUCT PROOF / 32.0–40.0

**Concept:** We are inside the real StructSynth interface, but only the relevant panes: seed questions, neighborhood slider, reasoning profile, result status. The product converts raw problems into a controllable structure experiment.

**On screen:** Real labels: `种子题目`, `语段邻域比例`, `思维难度画像`, `结构族命中率`. A slider moves from 35% to 45%; three slot chips align; result card says `同类 ≠ 同句`.

**Direction:** Vertical rebuild of the actual page's warm-paper / dark-green UI. Large 2x control. Cursor clicks, panel reacts. No invented controls or metrics. Exit by result card growing to fill frame.

## BEAT 6 — PAYOFF / 40.0–45.0

**Concept:** The result card becomes the end card. Mark and claim land with one decisive coral-to-mint sweep.

**On screen:** `同类题，不复制题面。` / `复制的是思维。` StructSynth mark + `本地数学结构实验台`.

**Direction:** Deep green field, spectral graph still breathing behind. Wordmark wipes in. Final line holds. Strong end-card layout with safe margins.

## Production Architecture

```
promo-structsynth/
├── index.html
├── BRIEF.md
├── DESIGN.md
├── SCRIPT.md
├── STORYBOARD.md
├── assets/
│   ├── vendor/gsap.min.js
│   ├── brand/mark.svg
│   └── sfx/
└── compositions/
    ├── beat-1-hook.html
    ├── beat-2-function.html
    ├── beat-3-engine.html
    ├── beat-4-cmo.html
    ├── beat-5-ui.html
    └── beat-6-payoff.html
```
