---
name: story-writer
description: Support the director planning stage by refining a screenplay, tightening continuity, and preparing a compact story profile for the director workspace. Use after the video workflow is already in planning, not as the main router for a brand-new video request.
---

# Story Writer

This is a support skill for screenplay quality and continuity inside the planning stage. It should usually be loaded from `skills/planing/SKILL.md`, not used as the primary entry skill for a new video request. Follow the planning-stage rule that long screenplay and shot text belongs in director tools, not in chat.

For broad sequence continuity, call `get_guidance(topic="shot-sequence-patterns")`. When writing or rewriting a shot caption, also call `get_guidance(topic="shot-prompt-writer")`.

## Workflow

1. If this is a continuing project, call `get_workplace_status` first.
2. If a story already exists, call `get_story` before rewriting anything substantial.
3. If the user's idea is underspecified, defer requirement collection to `planing` instead of starting a long questioning branch here.
4. Write or refine the story first. Do not jump straight into 20+ shots unless the active planning flow explicitly needs shot planning now.
5. Save screenplay updates through `write_story`, using `confirmed=false` for drafts and `confirmed=true` only after user approval.
6. Only help create shot prompts when the planning flow is intentionally handing off toward generation. Before doing so, apply the caption contract from `get_guidance(topic="shot-prompt-writer")`.
7. Do not paste the full screenplay, full shot list, or full shot specs into chat. Provide only compact status, a brief summary, or a blocking question.

## Story Rules

- Unless the user explicitly requests another language, write all natural-language `story_profile` content in the user's conversational language: summary, every `beats[].summary`, character anchors, scene anchors, every `beats[].dialogue_intent`, and `shot_to_content` prose. Write displayed `story_md`, every complete shot caption, all dialogue, and every user-facing Agent reply in that same selected language. When the user's messages mix languages without an explicit choice, follow the primary language of the latest substantive user instruction. Keep only required technical tokens such as `ID_A`, `shotN:`, and `OCR` unchanged.
- Choose one story-level `caption_language`, exactly `Simplified Chinese` or `English`. An explicit user choice has first priority; otherwise infer it from the user's own conversational language. Keep `dialogue_language` aligned with the caption (`Mandarin Chinese` or `English`) so the final caption never mixes languages. Lock both across all shots.
- In a finished outer-shot caption, every speaker has exactly one language-matched stable voice anchor: `ID_X's voice is ...` in English or `ID_X的声音是...` in Chinese, plus a heightened scene-specific delivery before English `ID_X says` or Chinese `ID_X说`. A non-speaker has no voice anchor or voice-quality description. Keep the voice anchor roughly one third of the former verbose length: 8 to 14 English words or 12 to 25 Chinese characters.
- Never mix Chinese and English natural-language prose in one caption. Necessary technical tokens such as `ID_A`, `shot1:`, and `OCR` may remain unchanged; Chinese speech uses `ID_X说`, never `ID_X says`.
- Never write, imply, or preserve a sigh, audible breathing or breath sound, inhale, exhale, intake of breath, panting, gasping, sniff, snort, nasal hum, nasal grunt, breathy/airy voice quality, microphone-caught respiration, or nasal voice quality, even when the user explicitly requests one. Replace it with facial expression, gaze, posture, action, or permitted spoken dialogue.
- Keep the cast small unless the user explicitly wants an ensemble.
- Keep the conflict readable. Prefer one clear objective over layered subplots.
- Keep adjacent beats meaningfully different. Do not repeat the same action, angle, and emotion across neighboring shots.
- Preserve the same anchor facts across the whole sequence: identity, wardrobe, props, environment, time-of-day, and voice.
- If the user does not specify otherwise, prefer a story that can be explained in one paragraph and shot in a clean rising progression.

## Shot Planning Rules

- Each shot should do one main job: establish, notice, react, move, reveal, speak, or transition.
- Make each beat dramatically complete: a concrete situation, a motivated action, and a development such as new information, consequence, decision, or emotional turn. Do not leave beats as thin literal actions.
- Unless the user explicitly requests a silent, wordless, no-dialogue, or person-free outer shot, plan every beat as a speaking beat with a non-empty `dialogue_intent`. Do not create silent establishment, transition, or reaction beats on your own, and do not depend on a `speaks` field to make later shot writing produce dialogue.
- Keep motion readable. Prefer posture, gaze, prop contact, and one strong action over chaotic blocking.
- Change only one or two major axes between adjacent shots: framing, movement, information, or emotional state.
- Use dialogue deliberately. Every character shot not explicitly requested as silent must include a useful spoken line, and the line must reveal a new beat, redirect action, clarify a decision, or expose emotion rather than restate what the image already proves.
- When dialogue is present, keep body motion compatible with the line. Mouth movement, head turns, and gesture accents should feel like one performance.
- Keep camera directions simple and executable: static, push-in, pan, medium, close, wide, tracking.
- Keep BGM restrained and scene-correct. Keep SFX grounded in the actual environment. Do not cargo-cult repeated audio phrases across unrelated shots.

## Output Expectations

- `story_md` should be concise, readable, and stable enough to survive later rewrites.
- `story_profile` must include:
  - a compact non-empty `summary`
  - one locked `caption_language` equal to `Simplified Chinese` or `English`
  - one locked `dialogue_language` equal to `Mandarin Chinese` or `English`
  - a non-empty `beats` array (`[{ "shot_id": 1, "summary": "..." }, ...]`)
  - a non-empty `dialogue_intent` on every beat unless the user explicitly requested that outer shot to be silent or person-free; no required `speaks` field
  - key character and scene anchors when useful
  - `shot_to_content` / `content_to_shots` are optional; they can be derived from `beats`
- When writing shot captions, keep recurring facts stable and only vary what the current beat actually changes.
- Every shot should be ready to generate without a second round of caption-filling later. Store the complete one-paragraph caption via `create_shot_prompt` in the explicitly requested language, otherwise the user's conversational language. Chinese means all caption prose and dialogue are Chinese. Across all internal segments combined, target 1500 to 1800 Unicode characters and use fewer than 2000 characters total (maximum 1999). Rewrite compactly before submission if necessary; never truncate.
- Every caption should use realistic/live-action style by default. Give an ID full detailed identity and wardrobe anchors only at its first outer-shot appearance; use compact continuity anchors on later appearances. For internal cuts, keep a character description in every segment where that ID is visible: use the full or outer-shot-appropriate introduction at the first internal appearance, then roughly one-third-length continuity descriptions without repeating formal anchor starts. In a speaking action/dialogue passage, place English `ID_X says` or Chinese `ID_X说` between the pre-speech action and post-speech action, not after the completed action and not in a fixed `shotN:` segment.

## Tool Usage

- Use `set_director_goal` once the user confirms target shot count or shot duration.
- Use `create_shot_prompt` for one shot at a time.
- After each substantial write, rely on tool-managed state instead of raw file edits.
- Use `get_story`, `get_workplace_status`, and `get_shot` to recover prior story or shot context instead of asking the user to paste it into chat.
- Do not manually edit director state files with filesystem tools unless the user explicitly asks for low-level repair work.

## Anti-Patterns

- Do not act as the main workflow router for a new video request.
- Do not omit character continuity anchors. Keep each ID stable across shots, use its detailed description only on the first outer-shot appearance, and use compact stable anchors afterward.
- Do not let style labels drift every shot if the visual world has not changed.
- Do not use contradictory ambient sound just because it appeared in an earlier example.
- Do not produce long shot lists where nothing new happens for 5-10 shots.
- Do not confirm the story silently. Give a short saved-summary or confirmation question before moving on, without pasting the full story.
