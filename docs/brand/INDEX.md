# vidtight brand pack

Locked 2026-11-19. Low-poly shears cutting an amber waveform. Informal: *the shears*. No character, no face, no byline under the headline.

| Piece | Call |
|---|---|
| Mark | Bone-ivory tailor's shears, late-1990s low-poly, both finger loops in frame |
| Cut | Amber waveform through the blades. The cut tail falls off as warm smoke |
| Hero | Shears enter from the right. Left side stays empty for type |
| Palette | Warm black `#0E0A07`, ivory shears, amber `#E8A85A`. Not Browser Hand's ink void or cyan |
| Type | Kicker `vidtight`. Headline **Tighten the take**. No third line |
| Voice | Quiet, precise. The headline is the promise; a feature list under it would date the card |

## Where each file goes

| Surface | File | Size |
|---|---|---|
| GitHub repo page (README hero) | `derived/github-social-1280x640.png` | 1280×640 |
| GitHub **Settings → Social preview** (link unfurls) | same file | 1280×640 |

The social-preview slot is a manual upload. The README link does not fill it.

## Masters

Editable HTML for the card lives in `src/`. The raster master is the type-free scene:

- `masters/banner-scene.jpg` — shears, waveform, and smoke, no letters

Re-render the card:

```bash
node ~/.autovault/skills/html-asset-renderer/scripts/render-html-assets.mjs \
  docs/brand/src docs/brand/derived \
  --manifest docs/brand/src/assets.manifest.json \
  --selector .canvas
```
