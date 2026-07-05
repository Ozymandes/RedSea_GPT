---
name: minimalist-ui
description: Clean editorial-style interfaces. Warm monochrome palette, typographic contrast, flat bento grids, muted pastels. No gradients, no heavy shadows.
source: https://github.com/Leonxlnx/taste-skill/blob/main/skills/minimalist-skill/SKILL.md
---

# Protocol: Premium Utilitarian Minimalism UI Architect

## 1. Protocol Overview
Name: Premium Utilitarian Minimalism & Editorial UI
Description: An advanced frontend engineering directive for generating highly refined, ultra-minimalist, "document-style" web interfaces analogous to top-tier workspace platforms. This protocol strictly enforces a high-contrast warm monochrome palette, bespoke typographic hierarchies, meticulous structural macro-whitespace, bento-grid layouts, and an ultra-flat component architecture with deliberate muted pastel accents. It actively rejects standard generic SaaS design trends.

## 2. Absolute Negative Constraints (Banned Elements)
The AI must strictly avoid the following generic web development defaults:
- DO NOT use the "Inter", "Roboto", or "Open Sans" typefaces.
- DO NOT use generic, thin-line icon libraries like "Lucide", "Feather", or standard "Heroicons".
- DO NOT use Tailwind's default heavy drop shadows (e.g., `shadow-md`, `shadow-lg`, `shadow-xl`). Shadows must be practically non-existent or heavily customized to be ultra-diffuse and low opacity (< 0.05).
- DO NOT use primary colored backgrounds for large elements or sections.
- DO NOT use gradients, neon colors, or 3D glassmorphism (beyond subtle navbar blurs).
- DO NOT use `rounded-full` (pill shapes) for large containers, cards, or primary buttons.
- DO NOT use emojis anywhere in code, markup, text content, headings, or alt text.
- DO NOT use AI copywriting clichés: "Elevate", "Seamless", "Unleash", "Next-Gen", "Game-changer", "Delve".

## 3. Typographic Architecture
- Primary Sans-Serif: `'SF Pro Display', 'Geist Sans', 'Helvetica Neue', 'Switzer', sans-serif`.
- Editorial Serif (Hero/Quotes): `'Lyon Text', 'Newsreader', 'Playfair Display', 'Instrument Serif', serif`. Tight tracking (`letter-spacing: -0.02em` to `-0.04em`), line-height `1.1`.
- Monospace: `'Geist Mono', 'SF Mono', 'JetBrains Mono', monospace`.
- Body text never absolute black. Use `#111111` or `#2F3437`, `line-height: 1.6`. Secondary muted gray `#787774`.

## 4. Color Palette (Warm Monochrome + Spot Pastels)
- Canvas: `#FFFFFF` or `#F7F6F3` / `#FBFBFA`.
- Cards: `#FFFFFF` or `#F9F9F8`.
- Borders: `#EAEAEA` or `rgba(0,0,0,0.06)`.
- Accents (desaturated pastels): Pale Red `#FDEBEC`/`#9F2F2D`, Pale Blue `#E1F3FE`/`#1F6C9F`, Pale Green `#EDF3EC`/`#346538`, Pale Yellow `#FBF3DB`/`#956400`.

## 5. Component Specs
- Bento grids: asymmetrical CSS Grid, cards `border: 1px solid #EAEAEA`, radius `8px`–`12px`, padding `24px`–`40px`.
- CTAs: solid `#111111`/`#FFFFFF`, radius `4px`–`6px`, no shadow, hover `#333333` or `scale(0.98)`.
- Tags/badges: pill (`9999px`), `text-xs`, uppercase, wide tracking, muted pastel bg.
- Accordions: strip containers, `border-bottom: 1px solid #EAEAEA`, `+`/`-` toggle.
- Keystrokes: `<kbd>`, `1px solid #EAEAEA`, radius `4px`, `#F7F6F3`, monospace.

## 6. Iconography & Imagery
- Icons: Phosphor (Bold/Fill) or Radix. Standardize stroke width.
- Photography: desaturated, warm tone, subtle `opacity 0.04` warm grain overlay.
- Hero bg: subtle low-opacity imagery, soft radial light spots (`opacity 0.03`), minimal geometric line patterns.

## 7. Subtle Motion
Motion should feel invisible — quiet sophistication, not spectacle.
- Scroll entry: `translateY(12px)` + `opacity: 0` over `600ms` `cubic-bezier(0.16, 1, 0.3, 1)`. Use `IntersectionObserver`.
- Hover: ultra-subtle shadow `0 2px 8px rgba(0,0,0,0.04)` over `200ms`. Buttons `scale(0.98)` on `:active`.
- Staggered reveals: `animation-delay: calc(var(--index) * 80ms)`.
- Ambient: optional single slow radial blob (`20s+`, `opacity 0.02–0.04`), `position: fixed; pointer-events: none`.
- Performance: only `transform`/`opacity`. `will-change: transform` sparingly.

## 8. Execution Protocol
1. Establish macro-whitespace first (`py-24`/`py-32`).
2. Constrain content width `max-w-4xl`/`max-w-5xl`.
3. Apply typographic hierarchy + monochrome vars immediately.
4. Every card/divider: `1px solid #EAEAEA`.
5. Scroll-entry animations on all major blocks.
6. Depth via imagery/ambient gradients/textures — no empty flat backgrounds.
7. Native high-end editorial aesthetic without manual adjustment.

## Red Sea adaptation (project-specific)
This skill bans primary colored backgrounds, but the Red Sea brand demands an ocean
palette. The adaptation: keep the STRUCTURE and CRAFT (editorial typography, 1px
borders, generous whitespace, restrained motion, off-black text) and treat ocean
deep-teal + coral as the SCARCE semantic color (used like the muted pastels — for
accents, tags, key data — never as a giant gradient hero). Deep teal = the "ink",
warm bone canvas = the "paper". This is taste: knowing which rules to keep (craft)
and which to adapt (palette) for a brand.
