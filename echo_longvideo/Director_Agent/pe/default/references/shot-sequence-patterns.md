# Shot Sequence Patterns

This reference distills the useful structure from a long single-subject seafaring sequence without copying its repetition.

## What The Example Did Well

- It locked one focal subject across many shots with a stable identity anchor.
- It advanced the story in small readable steps instead of giant jumps.
- It staged dialogue after a readable visual setup within the same shot.
- It used simple camera verbs that are easy for a shot-writing model to preserve.
- It kept the larger trajectory clear:
  - horizon
  - discovery
  - approach
  - landing
  - move inland

## What Not To Copy Literally

- Repeating bloated identity paragraphs wastes tokens. Every shot caption still needs compact but complete character anchors in every shot that shows people; use `shot-prompt-writer.md` for that stricter caption format.
- Repeating the same BGM and SFX boilerplate makes the sequence feel synthetic.
- Contradictory sound descriptions break immersion.
- Swapping style labels every shot without a real visual change creates noise, not control.

## Reusable Rules

### 1. Lock The Anchors

Across a shot run, keep these stable unless the story explicitly changes them:

- subject ID
- appearance and wardrobe
- voice identity
- core environment
- immediate objective

### 2. Move One Beat At A Time

Every shot must be a compact but dramatically complete unit, not a thin literal
description. Give it a concrete situation, a motivated action, and at least one
development or payoff. Each shot should add one of these:

- new information
- new intention
- new movement
- new obstacle
- new emotional turn

If a shot adds none of them, cut it or merge it.

Do not settle for a bare beat such as "looks at the door", "walks closer", or
"says one obvious sentence". Add specific performance, environmental, causal,
or emotional detail that makes the moment feel complete while preserving the
confirmed plot and keeping the action achievable within the shot duration.

### 3. Keep Adjacent Shots Distinct But Continuous

Between neighboring shots, vary one or two of:

- framing size
- camera motion
- body action
- information revealed
- speech state

Do not change all five at once without a hard narrative reason.

### 4. Use Dialogue As A Turning Point

Dialogue should appear when it does one of these:

- confirms a discovery
- redirects the group
- reveals attitude
- commits to the next action

Avoid filler lines that merely narrate the visible image.

### 5. Lock One Full-Content Language For The Whole Story

Choose `caption_language` and the aligned `dialogue_language` exactly once while
writing `story_profile`. Their values must be `Simplified Chinese` plus
`Mandarin Chinese`, or `English` plus `English`.

Use this priority:

1. An explicit user choice of language.
2. Otherwise, infer it from the user's own conversational language: primarily
   Chinese means the Chinese pair above; primarily English means the English pair.
3. If the user's conversation mixes Chinese and English without an explicit
   choice, use the primary language of the latest substantive user instruction.

Consider only the user's own conversational messages when applying this
fallback. Ignore quoted story dialogue, pasted captions, character names,
ethnicity, nationality, appearance, and location. Also ignore the English
language of PE references, system prompts, tool instructions, or previously
generated content. Once chosen, the language is locked across the screenplay,
story profile, every speaker, and every shot; later wording cannot switch it.

Unless the user explicitly requests another language, all natural-language
content follows the user's conversational language: displayed `story_md`,
`story_profile.summary`, every beat summary and dialogue intent, character and
scene anchors, semantic mapping prose, complete shot-caption prose, dialogue,
and user-facing Agent replies. Never mix Chinese and English natural-language
prose in one artifact. Technical tokens such as `ID_A`, `shot1:`, and `OCR`
may remain unchanged.

Plan speech at story level instead of letting each shot guess. Unless the user
explicitly requests a silent, wordless, no-dialogue, or person-free outer shot,
every beat must produce spoken dialogue and have a non-empty
`"dialogue_intent"` describing the new information, decision, redirection, or
emotional turn the line contributes. Do not invent silent establishment,
transition, or reaction beats. Do not require or rely on a `speaks` field: it
may be absent after story-profile normalization, but the default-speech rule
still applies.

In every finished outer-shot caption, each character who actually speaks must
have exactly one complete stable voice anchor beginning `ID_X's voice is ` and
describing register, timbre, resonance, accent, and articulation. A changing
delivery phrase before `ID_X says` must also match that shot's dialogue,
emotion, and scene with heightened, unmistakably dramatic intensity; it does
not replace the stable anchor. Use the full emotional range rather than making
every line calm or turning every line into a shout: fear may be tightly
panicked, grief close to breaking, joy explosively radiant, anger cutting and
volatile, suspicion controlled but dangerous, and tenderness fiercely
vulnerable. A character who does not speak must have no voice anchor and no
voice-quality description anywhere.

The complete caption for one outer shot should target 1500 to 1800 Unicode
characters and must contain fewer than 2000 characters total (maximum 1999)
across all internal segments combined. Count every letter or Han character,
space, punctuation mark, quote, `This video has...` prefix, `shotN:` label, and
required declaration. Roughly 220 to 280 English words is only a soft planning
guide; the character budget always wins. Compose concisely from the start. If a
draft is too long, rewrite and compress secondary environment, lighting,
composition, and camera modifiers before output while preserving required
anchors, voice and delivery, dialogue, vocal/OCR declarations, and segment
syntax. Never truncate a finished caption.

Never write, imply, or preserve a sigh, audible breathing or breath sound,
inhale, exhale, intake of breath, panting, gasping, sniff, snort, nasal hum,
nasal grunt, breathy or airy voice quality, microphone-caught respiration, or
nasal voice quality, even when the user explicitly requests one. Replace it
with facial expression, gaze, posture, action, or permitted spoken dialogue.
This prohibition applies while planning the story and while writing every shot
caption.

### 6. Build In Mini-Arcs

For long sequences, every 4-6 shots should create a local progression:

- setup -> notice -> react -> move
- scan -> reveal -> decide -> act
- approach -> inspect -> conclude -> commit

This prevents the middle of the sequence from flattening out.

## Recommended Story Profile Shape

When writing `story_profile`, keep it compact and tool-friendly. The English
sample below illustrates structure only; translate every natural-language value
to the selected Chinese language when the user is communicating in Chinese:

```json
{
  "summary": "Short story summary",
  "dialogue_language": "English",
  "beats": [
    {
      "shot_id": 1,
      "summary": "The hero reaches the rain-soaked deck and discovers the navigation lights are dead.",
      "dialogue_intent": "Reports the failed navigation lights and asks the crew to check the backup circuit."
    },
    {
      "shot_id": 2,
      "summary": "The hero recognizes a signal on the horizon and commits to following it.",
      "dialogue_intent": "Names the signal and commits the crew to the approach."
    },
    {
      "shot_id": 3,
      "summary": "The hero reaches the transmitter and realizes the signal carries a familiar voice.",
      "dialogue_intent": "Identifies the voice and reveals why the signal matters personally."
    }
  ],
  "anchors": {
    "characters": ["Stable identity facts"],
    "environment": ["Stable world facts"],
    "tone": ["Stable tone facts"]
  },
  "shot_to_content": {
    "shot_001": "establishes the hero on deck",
    "shot_002": "hero notices the horizon",
    "shot_003": "hero recognizes the transmitted voice"
  },
  "content_to_shots": {
    "signal discovery": ["shot_002", "shot_003"]
  }
}
```

Keep the mappings semantic. They are for lookup, not for prose.
