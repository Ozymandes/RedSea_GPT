# Design skills (installed for the Red Sea GPT web demo)

These encode the craft and anti-slop rules for the demo frontend. They are
reference material for building the Vite + React UI; they do not affect the
Python RAG backend.

## Installed

| Skill | Source | Role |
|---|---|---|
| `emil-design-eng` | emilkowalski/skill | Animation decision framework, motion craft, "unseen details compound". The motion bible. |
| `minimalist-ui` | Leonxlnx/taste-skill | Editorial / Linear-Notion aesthetic: typographic contrast, 1px borders, restrained palette. Base visual language. |
| `review-animations` | emilkowalski/skill | Animation review checklist + standards. |
| `animation-vocabulary` | emilkowalski/skill | Motion vocabulary for prompting. |

## Not yet pulled (referenced, optional)
- **Impeccable** (`pbakaus/impeccable`): 45 deterministic anti-pattern detector
  rules + 23 commands. The anti-slop linter. Install via `npx impeccable install`
  when working in an interactive terminal; the key rules overlap with
  `minimalist-ui`'s "Banned Elements" section (no Inter/Roboto, no `shadow-md`,
  no `rounded-full` on containers, no big gradients, no emojis, no AI clichés).

## Design language for the Red Sea demo (synthesis)
Taste = trained restraint. The three skills agree on the craft; the brand
demands one adaptation (ocean palette instead of warm monochrome).

- **Mood:** calm, deep, scientific. Not bouncy. Not flashy. Deliberate.
- **Canvas:** warm bone (`#F7F6F3`) "paper"; deep teal `#0E3B43` as the "ink";
  coral `#E07856` as the single scarce semantic accent (citations, key data).
- **Type:** editorial serif for hero/headings, clean sans for body, mono for
  metadata + reasoning trace.
- **Motion:** only `transform`/`opacity`. UI under 300ms. Custom
  `cubic-bezier(0.23, 1, 0.32, 1)` ease-out. `scale(0.97)` on press. Never
  `scale(0)`. Reasoning trace animates sparingly — it is informational, not
  decorative.
- **Borders:** `1px solid #EAEAEA`. No heavy shadows.
- **Anti-slop:** no emojis, no "Elevate/Seamless/Unleash", no generic Lucide-only
  icons (Phosphor/Radix), no gradient hero blocks.
