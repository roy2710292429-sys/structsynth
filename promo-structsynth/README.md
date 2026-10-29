# StructSynth Promo

45-second HyperFrames promo for StructSynth, rendered from HTML.

## Specs

- 1080×1920 portrait
- 30 fps
- 45 seconds
- No human voice or avatar
- Six scenes
- Local GSAP runtime and local media assets

## Preview

```bash
npm install
npm run dev
```

## Validate

```bash
npx hyperframes@0.8.210 lint
```

`check`, `snapshot`, `preview`, and `render` additionally require a Chromium
runtime. In this repository's original creation environment, the macOS Codex
sandbox blocked Chromium's Mach-port bootstrap with `SIGABRT`; the checked-in
project is ready to render on a normal desktop environment.

## Render

```bash
npx hyperframes@0.8.210 render \
  --quality delivery \
  --output renders/structsynth-promo.mp4
```

FFmpeg and FFprobe must be installed. Cloud rendering requires HeyGen login:

```bash
npx hyperframes@0.8.210 auth login
npx hyperframes@0.8.210 cloud render \
  --quality high \
  --aspect-ratio 9:16 \
  --output renders/structsynth-promo.mp4
```

## Project Map

```text
index.html                  Root timeline and SFX placement
compositions/               One file per scene
assets/brand/               StructSynth mark
assets/sfx/                 Offline sound effects and Pixabay credits
assets/vendor/gsap.min.js   Pinned local animation runtime
BRIEF.md                    Creative brief
DESIGN.md                   Color/font/component reference
SCRIPT.md                   On-screen copy and timing
STORYBOARD.md               Scene-by-scene direction
REVIEW.md                   Validation and known render blocker
```
