/** Build the persisted WebUI session key (mirrors ``webui_session_key`` in Python). */
export function webuiSessionKey(
  userId: string | null | undefined,
  chatId: string,
): string {
  const uid = userId?.trim();
  const cid = chatId.trim();
  if (!uid) {
    throw new Error("webuiSessionKey requires userId");
  }
  if (!cid) {
    throw new Error("webuiSessionKey requires chatId");
  }
  return `websocket:${uid}:${cid}`;
}

/** Return true for deprecated ``websocket:<chatId>`` keys (no user segment). */
export function isLegacyTwoPartSessionKey(key: string): boolean {
  if (!key.startsWith("websocket:")) return false;
  const rest = key.slice("websocket:".length);
  return rest.length > 0 && !rest.includes(":");
}

/** Extract the wire ``chat_id`` from a persisted session key. */
export function webuiChatIdFromKey(key: string): string {
  return key.slice(key.lastIndexOf(":") + 1);
}
