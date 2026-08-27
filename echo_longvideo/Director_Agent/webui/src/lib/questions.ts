import { splitMediaByKind } from "@/lib/media";
import type { SessionWireMessage } from "@/lib/api";
import type { UIQuestionCard, UIMessage, WireQuestionCard } from "@/lib/types";

const BATCH_ID_RE =
  /batch=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i;

const AGENT_INJECT_STARTS = [
  "REFERENCE_IMAGE_GATE ",
  "STORY_DIRECTION_CONFIRM ",
] as const;

/** Drop LLM-only inject prefixes so the thread shows the user's card answer. */
export function visibleUserContent(content: string): string {
  const stripped = content.trim();
  if (!AGENT_INJECT_STARTS.some((prefix) => stripped.startsWith(prefix))) {
    return content;
  }
  const parts = stripped.split(/\n\n/, 2);
  if (parts.length === 2 && parts[1]?.trim()) {
    return parts[1].trim();
  }
  const lines = stripped
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length >= 2 ? (lines[lines.length - 1] ?? "") : "";
}

function withVisibleUserContent(
  messages: SessionWireMessage[],
): SessionWireMessage[] {
  return messages.map((row) => {
    if (row.role !== "user" || typeof row.content !== "string") return row;
    const visible = visibleUserContent(row.content);
    return visible === row.content ? row : { ...row, content: visible };
  });
}

/** Normalize backend / session question cards into UI shape. */
export function wireQuestionCards(raw: WireQuestionCard[]): UIQuestionCard[] {
  return raw.flatMap((card, index): UIQuestionCard[] => {
    if (!card || typeof card !== "object") return [];
    const question =
      typeof card.question === "string" ? card.question.trim() : "";
    if (!question || !Array.isArray(card.options)) return [];

    const options: { label: string }[] = [];
    for (const opt of card.options) {
      if (typeof opt === "string" && opt.trim()) {
        options.push({ label: opt.trim() });
        continue;
      }
      if (
        typeof opt === "object" &&
        opt !== null &&
        typeof opt.label === "string" &&
        opt.label.trim()
      ) {
        options.push({ label: opt.label.trim() });
      }
    }
    if (options.length === 0) return [];

    const allowCustom =
      card.allow_custom === true || card.allowCustom === true;
    const id =
      typeof card.id === "string" && card.id.trim()
        ? card.id.trim()
        : `q-${index}`;
    return [
      {
        id,
        question,
        options,
        ...(allowCustom ? { allowCustom: true } : {}),
        ...(card.answered != null ? { answered: card.answered } : {}),
      },
    ];
  });
}

/** Extract question batch id from an ask_user tool result string. */
export function extractBatchIdFromToolResult(
  content: string,
): string | undefined {
  const match = content.match(BATCH_ID_RE);
  return match?.[1];
}

/** Parse one OpenAI-style tool_call for ask_user payload. */
export function parseAskUserToolCall(toolCall: unknown): {
  toolCallId: string;
  content: string;
  questions: WireQuestionCard[];
} | null {
  if (!toolCall || typeof toolCall !== "object") return null;
  const tc = toolCall as {
    id?: unknown;
    function?: { name?: unknown; arguments?: unknown };
  };
  if (typeof tc.id !== "string" || !tc.id.trim()) return null;
  const fn = tc.function;
  if (!fn || fn.name !== "ask_user") return null;
  if (typeof fn.arguments !== "string" || !fn.arguments.trim()) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(fn.arguments);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const args = parsed as {
    content?: unknown;
    questions?: unknown;
  };
  if (!Array.isArray(args.questions) || args.questions.length === 0) {
    return null;
  }
  const intro =
    typeof args.content === "string" ? args.content.trim() : "";

  const questions: WireQuestionCard[] = [];
  for (const item of args.questions) {
    if (!item || typeof item !== "object") continue;
    const card = item as WireQuestionCard;
    if (typeof card.question !== "string" || !Array.isArray(card.options)) {
      continue;
    }
    questions.push({
      id: typeof card.id === "string" ? card.id : `q-${questions.length}`,
      question: card.question,
      options: card.options,
      ...(card.allow_custom === true || card.allowCustom === true
        ? { allow_custom: true }
        : {}),
      ...(card.answered != null ? { answered: card.answered } : {}),
    });
  }
  if (questions.length === 0) return null;

  return {
    toolCallId: tc.id,
    content: intro,
    questions,
  };
}

/** Index tool_call_id → batch_id from raw session tool results. */
export function indexAskUserBatches(
  messages: Array<{
    role: string;
    tool_call_id?: string;
    name?: string;
    content?: string;
  }>,
): Map<string, string> {
  const index = new Map<string, string>();
  for (const m of messages) {
    if (m.role !== "tool") continue;
    if (typeof m.tool_call_id !== "string" || !m.tool_call_id.trim()) {
      continue;
    }
    const isAskUser =
      m.name === "ask_user" ||
      (typeof m.content === "string" &&
        m.content.includes("Question cards sent") &&
        m.content.includes("batch="));
    if (!isAskUser || typeof m.content !== "string") continue;
    const batchId = extractBatchIdFromToolResult(m.content);
    if (batchId) index.set(m.tool_call_id, batchId);
  }
  return index;
}

function allAskUserToolCalls(
  toolCalls: unknown,
): NonNullable<ReturnType<typeof parseAskUserToolCall>>[] {
  if (!Array.isArray(toolCalls)) return [];
  const parsed: NonNullable<ReturnType<typeof parseAskUserToolCall>>[] = [];
  for (const tc of toolCalls) {
    const item = parseAskUserToolCall(tc);
    if (item) parsed.push(item);
  }
  return parsed;
}

function firstAskUserToolCall(
  toolCalls: unknown,
): ReturnType<typeof parseAskUserToolCall> {
  const all = allAskUserToolCalls(toolCalls);
  return all[0] ?? null;
}

function mergeAskUserQuestions(
  toolCalls: unknown,
): WireQuestionCard[] {
  const merged: WireQuestionCard[] = [];
  for (const item of allAskUserToolCalls(toolCalls)) {
    for (const card of item.questions) {
      merged.push(card);
    }
  }
  return merged;
}

function assistantHasQuestions(
  row: Pick<SessionWireMessage, "role" | "questions" | "tool_calls">,
): boolean {
  if (row.role !== "assistant") return false;
  if (Array.isArray(row.questions) && row.questions.length > 0) return true;
  return mergeAskUserQuestions(row.tool_calls).length > 0;
}

function cardAcceptsReply(card: UIQuestionCard, reply: string): boolean {
  if (card.options.some((opt) => opt.label === reply)) {
    return true;
  }
  return card.allowCustom === true;
}

function assistantBatchAcceptsReply(
  questions: UIQuestionCard[],
  reply: string,
): boolean {
  const pending = questions.filter((q) => q.answered == null);
  if (pending.length === 0) {
    return false;
  }
  return pending.some((q) => cardAcceptsReply(q, reply));
}

function replyMatchesKnownCardAnswer(
  messages: SessionWireMessage[],
  text: string,
  batchByToolCallId: Map<string, string>,
): boolean {
  for (const m of messages) {
    if (m.role !== "assistant") continue;
    const { questions } = wireQuestionsFromSessionRow(m, batchByToolCallId);
    if (!questions?.length) continue;
    if (
      questions.some(
        (q) =>
          q.answered === text ||
          q.options.some((opt) => opt.label === text),
      )
    ) {
      return true;
    }
  }
  return false;
}

/** Drop card-style user rows piled at the end or duplicated after refresh. */
function shouldDropMisplacedCardUserReply(
  messages: SessionWireMessage[],
  idx: number,
  batchByToolCallId: Map<string, string>,
  seenUserReplies: Set<string>,
): boolean {
  const row = messages[idx];
  if (row.role !== "user") return false;
  const text = typeof row.content === "string" ? row.content.trim() : "";
  if (!text) return true;
  if (seenUserReplies.has(text)) return true;

  if (!replyMatchesKnownCardAnswer(messages, text, batchByToolCallId)) {
    seenUserReplies.add(text);
    return false;
  }

  let nearestCardIdx: number | null = null;
  for (let i = idx - 1; i >= 0; i--) {
    const prior = messages[i];
    if (prior.role === "user") break;
    if (prior.role === "assistant" && assistantHasQuestions(prior)) {
      nearestCardIdx = i;
      break;
    }
  }

  if (nearestCardIdx === null) {
    seenUserReplies.add(text);
    return false;
  }

  const { questions } = wireQuestionsFromSessionRow(
    messages[nearestCardIdx],
    batchByToolCallId,
  );
  if (!questions?.length) {
    seenUserReplies.add(text);
    return false;
  }

  const matchesNearest =
    questions.some((q) => q.answered === text) ||
    assistantBatchAcceptsReply(questions, text);

  if (matchesNearest) {
    seenUserReplies.add(text);
    return false;
  }

  return true;
}

function cardBatchDisplayReply(questions: UIQuestionCard[]): string | null {
  for (const q of questions) {
    if (typeof q.answered === "string" && q.answered.trim()) {
      return q.answered.trim();
    }
  }
  return null;
}

function hasProperUserReplyAfterCard(
  messages: SessionWireMessage[],
  cardIdx: number,
  reply: string,
): boolean {
  for (let i = cardIdx + 1; i < messages.length; i++) {
    const row = messages[i];
    if (row.role === "user") {
      const text = typeof row.content === "string" ? row.content.trim() : "";
      if (text !== reply) continue;

      let nearestCardIdx: number | null = null;
      for (let j = i - 1; j >= 0; j--) {
        const prior = messages[j];
        if (prior.role === "user") break;
        if (prior.role === "assistant" && assistantHasQuestions(prior)) {
          nearestCardIdx = j;
          break;
        }
      }
      return nearestCardIdx === cardIdx;
    }
    if (row.role === "assistant" && assistantHasQuestions(row)) {
      continue;
    }
  }
  return false;
}

export function resolveQuestionCardsFromThread(
  messages: UIMessage[],
): UIMessage[] {
  const next = messages.map((m) => ({
    ...m,
    questions: m.questions?.map((q) => ({ ...q })),
  }));

  for (let userIdx = 0; userIdx < next.length; userIdx++) {
    const row = next[userIdx];
    if (row.role !== "user") continue;
    const reply = row.content.trim();
    if (!reply) continue;

    for (let i = userIdx - 1; i >= 0; i--) {
      const prior = next[i];
      if (prior.role !== "assistant" || !prior.questions?.length) continue;
      if (!assistantBatchAcceptsReply(prior.questions, reply)) continue;

      next[i] = {
        ...prior,
        questions: prior.questions.map((q) =>
          q.answered == null && cardAcceptsReply(q, reply)
            ? { ...q, answered: reply }
            : q,
        ),
      };
      break;
    }
  }

  return next;
}

function questionFingerprint(
  questions: UIQuestionCard[],
  questionBatchId?: string,
): string {
  const body = questionBodyFingerprint(questions);
  return questionBatchId ? `${questionBatchId}\0${body}` : body;
}

function questionBodyFingerprint(questions: UIQuestionCard[]): string {
  return questions
    .map((q) => `${q.id}\0${q.question.trim()}`)
    .filter(Boolean)
    .join("\n");
}

function wireQuestionsFromSessionRow(
  m: SessionWireMessage,
  batchByToolCallId: Map<string, string>,
): { questions?: UIQuestionCard[]; questionBatchId?: string } {
  let questions: UIQuestionCard[] | undefined;
  let questionBatchId: string | undefined;

  if (Array.isArray(m.questions) && m.questions.length > 0) {
    const wired = wireQuestionCards(m.questions);
    if (wired.length > 0) {
      questions = wired;
      if (typeof m.question_batch_id === "string" && m.question_batch_id.trim()) {
        questionBatchId = m.question_batch_id;
      }
    }
  }

  if (!questions?.length && m.role === "assistant" && m.tool_calls) {
    const askUser = firstAskUserToolCall(m.tool_calls);
    const mergedQuestions = mergeAskUserQuestions(m.tool_calls);
    const wired = wireQuestionCards(
      mergedQuestions.length > 0
        ? mergedQuestions
        : (askUser?.questions ?? []),
    );
    if (wired.length > 0) {
      questions = wired;
      questionBatchId =
        (typeof m.question_batch_id === "string" && m.question_batch_id.trim()
          ? m.question_batch_id
          : undefined) ??
        (askUser ? batchByToolCallId.get(askUser.toolCallId) : undefined);
    }
  }

  return { questions, questionBatchId };
}

function isPlainTextDuplicateOfNearbyQuestionCard(
  messages: SessionWireMessage[],
  idx: number,
  seenQuestionBatches: Set<string>,
  seenQuestionFingerprints: Set<string>,
  seenQuestionBodies: Set<string>,
): boolean {
  const row = messages[idx];
  if (row.role !== "assistant" || assistantHasQuestions(row)) {
    return false;
  }
  if (row.tool_calls?.length) {
    return false;
  }
  const text = typeof row.content === "string" ? row.content.trim() : "";
  if (!text) {
    return false;
  }

  for (let i = idx - 1; i >= Math.max(0, idx - 12); i--) {
    const prior = messages[i];
    if (prior.role === "user") {
      return false;
    }
    if (prior.role !== "assistant" || !assistantHasQuestions(prior)) {
      continue;
    }
    const batchByToolCallId = indexAskUserBatches(messages);
    const { questions, questionBatchId } = wireQuestionsFromSessionRow(
      prior,
      batchByToolCallId,
    );
    if (!questions?.length) {
      return false;
    }
    const parts = questions.map((q) => q.question.trim()).filter(Boolean);
    if (!(text === parts.join("\n") || parts.some((part) => text.includes(part)))) {
      continue;
    }
    const body = questionBodyFingerprint(questions);
    if (seenQuestionBodies.has(body)) {
      return true;
    }
    if (questionBatchId && seenQuestionBatches.has(questionBatchId)) {
      return true;
    }
    const fingerprint = questionFingerprint(questions, questionBatchId);
    if (fingerprint && seenQuestionFingerprints.has(fingerprint)) {
      return true;
    }
    return false;
  }
  return false;
}

function recoverQuestionCardFromNearbyPlainText(
  messages: SessionWireMessage[],
  idx: number,
  batchByToolCallId: Map<string, string>,
  seenQuestionBatches: Set<string>,
  seenQuestionFingerprints: Set<string>,
  seenQuestionBodies: Set<string>,
): UIMessage[] {
  const row = messages[idx];
  if (row.role !== "assistant" || assistantHasQuestions(row)) {
    return [];
  }
  if (row.tool_calls?.length) {
    return [];
  }
  const text = typeof row.content === "string" ? row.content.trim() : "";
  if (!text) {
    return [];
  }

  for (let i = idx - 1; i >= Math.max(0, idx - 12); i--) {
    const prior = messages[i];
    if (prior.role === "user") {
      return [];
    }
    if (prior.role !== "assistant" || !assistantHasQuestions(prior)) {
      continue;
    }
    const { questions, questionBatchId } = wireQuestionsFromSessionRow(
      prior,
      batchByToolCallId,
    );
    if (!questions?.length) {
      return [];
    }
    const parts = questions.map((q) => q.question.trim()).filter(Boolean);
    if (!(text === parts.join("\n") || parts.some((part) => text.includes(part)))) {
      continue;
    }
    const body = questionBodyFingerprint(questions);
    if (seenQuestionBodies.has(body)) {
      return [];
    }
    if (questionBatchId && seenQuestionBatches.has(questionBatchId)) {
      return [];
    }
    const fingerprint = questionFingerprint(questions, questionBatchId);
    if (fingerprint && seenQuestionFingerprints.has(fingerprint)) {
      return [];
    }
    if (questionBatchId) {
      seenQuestionBatches.add(questionBatchId);
    }
    if (fingerprint) {
      seenQuestionFingerprints.add(fingerprint);
    }
    seenQuestionBodies.add(body);
    return [
      {
        id: `hist-${idx}`,
        role: "assistant",
        content: "",
        createdAt: row.timestamp ? Date.parse(row.timestamp) : Date.now(),
        questions,
        ...(questionBatchId ? { questionBatchId } : {}),
      },
    ];
  }
  return [];
}

/** Drop assistant prose that re-lists options from an already-answered card. */
function isPlainTextRedundantQuestionListing(
  messages: SessionWireMessage[],
  idx: number,
): boolean {
  const row = messages[idx];
  if (row.role !== "assistant" || assistantHasQuestions(row)) {
    return false;
  }
  if (row.tool_calls?.length) {
    return false;
  }
  const text = typeof row.content === "string" ? row.content.trim() : "";
  if (!text) {
    return false;
  }

  const batchByToolCallId = indexAskUserBatches(messages);
  for (let i = 0; i < messages.length; i++) {
    if (i === idx) continue;
    const other = messages[i];
    if (other.role !== "assistant") continue;
    const { questions } = wireQuestionsFromSessionRow(other, batchByToolCallId);
    if (!questions?.length) continue;

    const labels = questions
      .flatMap((q) => q.options.map((o) => o.label))
      .filter(Boolean);
    const matchCount = labels.filter((label) => text.includes(label)).length;
    const question = questions[0]?.question ?? "";
    const matchesQuestion =
      question.length > 0 &&
      text.includes(question) &&
      matchCount >= 1;
    if (matchCount < 2 && !matchesQuestion) {
      continue;
    }

    const answered = questions.some((q) => q.answered != null);
    let userRepliedAfter = false;
    for (let j = i + 1; j < idx; j++) {
      if (messages[j]?.role === "user") {
        const reply =
          typeof messages[j].content === "string"
            ? messages[j].content.trim()
            : "";
        if (reply) {
          userRepliedAfter = true;
          break;
        }
      }
    }
    if (answered || userRepliedAfter) {
      return true;
    }
  }
  return false;
}

function isAskUserFollowupBlurb(
  messages: SessionWireMessage[],
  idx: number,
): boolean {
  const row = messages[idx];
  if (row.role !== "assistant" || assistantHasQuestions(row)) {
    return false;
  }
  if (typeof row.content !== "string" || !row.content.trim()) {
    return false;
  }
  if (row.tool_calls?.length) {
    return false;
  }
  let prev = idx - 1;
  while (prev >= 0) {
    const prior = messages[prev];
    if (prior.role === "user") {
      return false;
    }
    if (prior.role === "tool") {
      prev -= 1;
      continue;
    }
    if (prior.role === "assistant") {
      if (!assistantHasQuestions(prior)) {
        prev -= 1;
        continue;
      }
      for (let i = prev + 1; i < idx; i++) {
        if (messages[i]?.role === "user") {
          return false;
        }
      }
      return true;
    }
    prev -= 1;
  }
  return false;
}

/** True when a live UI assistant row is redundant text right after question cards. */
export function isUiAskUserFollowupBlurb(
  messages: UIMessage[],
  idx: number,
): boolean {
  const row = messages[idx];
  if (row.role !== "assistant" || (row.questions?.length ?? 0) > 0) {
    return false;
  }
  if (!row.content.trim()) {
    return false;
  }
  let prev = idx - 1;
  while (prev >= 0) {
    const prior = messages[prev];
    if (prior.role === "user") {
      return false;
    }
    if (prior.role === "assistant") {
      if (
        prior.turnWaiting &&
        !prior.content.trim() &&
        !(prior.questions?.length ?? 0)
      ) {
        prev -= 1;
        continue;
      }
      if (!(prior.questions?.length ?? 0)) {
        prev -= 1;
        continue;
      }
      for (let i = prev + 1; i < idx; i++) {
        if (messages[i]?.role === "user") {
          return false;
        }
      }
      return true;
    }
    prev -= 1;
  }
  return false;
}

/** Drop assistant blurbs that duplicate ask_user cards (live thread or replay). */
export function stripAskUserFollowupBlurbs(
  messages: UIMessage[],
): UIMessage[] {
  return messages.filter((_, idx) => !isUiAskUserFollowupBlurb(messages, idx));
}

/** Map persisted session messages into thread UI messages. */
export function wireSessionMessages(
  rawMessages: SessionWireMessage[],
): UIMessage[] {
  const messages = withVisibleUserContent(rawMessages);
  const batchByToolCallId = indexAskUserBatches(messages);
  const seenQuestionBatches = new Set<string>();
  const seenQuestionFingerprints = new Set<string>();
  const seenQuestionBodies = new Set<string>();
  const seenUserReplies = new Set<string>();

  const ui = messages.flatMap((m, idx) => {
    if (m.role !== "user" && m.role !== "assistant") return [];
    if (typeof m.content !== "string") return [];

    if (m.role === "user") {
      const text = m.content.trim();
      if (!text) return [];
      if (
        shouldDropMisplacedCardUserReply(
          messages,
          idx,
          batchByToolCallId,
          seenUserReplies,
        )
      ) {
        return [];
      }
    }

    if (isAskUserFollowupBlurb(messages, idx)) return [];
    if (isPlainTextRedundantQuestionListing(messages, idx)) return [];
    if (
      isPlainTextDuplicateOfNearbyQuestionCard(
        messages,
        idx,
        seenQuestionBatches,
        seenQuestionFingerprints,
        seenQuestionBodies,
      )
    ) {
      return [];
    }

    const recovered = recoverQuestionCardFromNearbyPlainText(
      messages,
      idx,
      batchByToolCallId,
      seenQuestionBatches,
      seenQuestionFingerprints,
      seenQuestionBodies,
    );
    if (recovered.length > 0) {
      return recovered;
    }

    const { images, videos } = splitMediaByKind(m.media_urls);

    const { questions, questionBatchId } = wireQuestionsFromSessionRow(
      m,
      batchByToolCallId,
    );

    if (questionBatchId && seenQuestionBatches.has(questionBatchId)) {
      return [];
    }
    const fingerprint = questions?.length
      ? questionFingerprint(questions, questionBatchId)
      : "";
    const body = questions?.length ? questionBodyFingerprint(questions) : "";
    if (body && seenQuestionBodies.has(body)) {
      return [];
    }
    if (fingerprint && seenQuestionFingerprints.has(fingerprint)) {
      return [];
    }
    if (questionBatchId) {
      seenQuestionBatches.add(questionBatchId);
    }
    if (fingerprint) {
      seenQuestionFingerprints.add(fingerprint);
    }
    if (body) {
      seenQuestionBodies.add(body);
    }

    // Cards carry the prompt; avoid duplicating it as markdown text above.
    const displayContent = questions?.length ? "" : m.content.trim();

    if (!displayContent && !questions?.length && !images && !videos) {
      return [];
    }

    const cardMsg: UIMessage = {
      id: `hist-${idx}`,
      role: m.role as UIMessage["role"],
      content: displayContent,
      createdAt: m.timestamp ? Date.parse(m.timestamp) : Date.now(),
      ...(images ? { images } : {}),
      ...(videos ? { videos } : {}),
      ...(questions?.length ? { questions } : {}),
      ...(questionBatchId ? { questionBatchId } : {}),
    };

    const batchReply =
      m.role === "assistant" && questions?.length
        ? cardBatchDisplayReply(questions)
        : null;
    if (
      batchReply &&
      !hasProperUserReplyAfterCard(messages, idx, batchReply)
    ) {
      seenUserReplies.add(batchReply);
      return [
        cardMsg,
        {
          id: `hist-${idx}-reply`,
          role: "user" as UIMessage["role"],
          content: batchReply,
          createdAt: m.timestamp ? Date.parse(m.timestamp) + 1 : Date.now(),
        },
      ];
    }

    return [cardMsg];
  });
  return resolveQuestionCardsFromThread(ui);
}

