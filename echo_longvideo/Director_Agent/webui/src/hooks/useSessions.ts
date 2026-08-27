import { useCallback, useEffect, useRef, useState } from "react";

import { useClient } from "@/providers/ClientProvider";
import i18n from "@/i18n";
import {
  ApiError,
  deleteSession as apiDeleteSession,
  fetchSessionMessages,
  listSessions,
} from "@/lib/api";
import { deriveTitle } from "@/lib/format";
import { wireSessionMessages } from "@/lib/questions";
import { webuiSessionKey } from "@/lib/session-key";
import type { ChatSummary, UIMessage } from "@/lib/types";

const EMPTY_MESSAGES: UIMessage[] = [];

/** Merge server rows with local-only chats that are not persisted yet. */
export function mergeSessionLists(
  prev: ChatSummary[],
  serverRows: ChatSummary[],
  localKeys: Set<string>,
): ChatSummary[] {
  const serverByKey = new Map(serverRows.map((row) => [row.key, row]));

  for (const key of serverByKey.keys()) {
    localKeys.delete(key);
  }

  const locals = prev.filter(
    (session) =>
      localKeys.has(session.key) && !serverByKey.has(session.key),
  );

  const sortedServer = [...serverRows].sort(
    (a, b) =>
      Date.parse(b.updatedAt ?? "") - Date.parse(a.updatedAt ?? ""),
  );

  return [...locals, ...sortedServer];
}

/** Sidebar state: fetches the full session list and exposes create / delete actions. */
export function useSessions(): {
  sessions: ChatSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  createChat: (options?: { autoGenerate?: boolean }) => Promise<string>;
  deleteChat: (key: string) => Promise<void>;
} {
  const { client, token, webuiUserId } = useClient();
  const [sessions, setSessions] = useState<ChatSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const tokenRef = useRef(token);
  tokenRef.current = token;
  const localSessionKeysRef = useRef(new Set<string>());

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const rows = await listSessions(tokenRef.current);
      setSessions((prev) =>
        mergeSessionLists(prev, rows, localSessionKeysRef.current),
      );
      setError(null);
    } catch (e) {
      const msg =
        e instanceof ApiError ? `HTTP ${e.status}` : (e as Error).message;
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const createChat = useCallback(
    async (options?: { autoGenerate?: boolean }): Promise<string> => {
      const chatId = await client.newChat(5_000, options);
      const key = webuiSessionKey(webuiUserId, chatId);
      localSessionKeysRef.current.add(key);
      setSessions((prev) => [
        {
          key,
          channel: "websocket",
          chatId,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          preview: "",
          source: "stepwise",
          autoGenerate: Boolean(options?.autoGenerate),
        },
        ...prev.filter((s) => s.key !== key),
      ]);
      return chatId;
    },
    [client, webuiUserId],
  );

  const deleteChat = useCallback(async (key: string) => {
    await apiDeleteSession(tokenRef.current, key);
    localSessionKeysRef.current.delete(key);
    setSessions((prev) => prev.filter((s) => s.key !== key));
  }, []);

  return { sessions, loading, error, refresh, createChat, deleteChat };
}

/** Lazy-load a session's on-disk messages the first time the UI displays it. */
export function useSessionHistory(key: string | null): {
  messages: UIMessage[];
  loading: boolean;
  error: string | null;
} {
  const { token } = useClient();
  const [state, setState] = useState<{
    key: string | null;
    messages: UIMessage[];
    loading: boolean;
    error: string | null;
  }>({
    key: null,
    messages: [],
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (!key) {
      setState({
        key: null,
        messages: [],
        loading: false,
        error: null,
      });
      return;
    }
    let cancelled = false;
    // Mark the new key as loading immediately so callers never see stale
    // messages from the previous session during the render right after a switch.
    setState({
      key,
      messages: [],
      loading: true,
      error: null,
    });
    (async () => {
      try {
        const body = await fetchSessionMessages(token, key);
        if (cancelled) return;
        const ui: UIMessage[] = wireSessionMessages(body.messages);
        setState({
          key,
          messages: ui,
          loading: false,
          error: null,
        });
      } catch (e) {
        if (cancelled) return;
        // A 404 just means the session hasn't been persisted yet (brand-new
        // chat, first message not sent). That's a normal state, not an error.
        if (e instanceof ApiError && e.status === 404) {
          setState({
            key,
            messages: [],
            loading: false,
            error: null,
          });
        } else {
          setState({
            key,
            messages: [],
            loading: false,
            error: (e as Error).message,
          });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [key, token]);

  if (!key) {
    return { messages: EMPTY_MESSAGES, loading: false, error: null };
  }

  // Even before the effect above commits its loading state, never surface the
  // previous session's payload for a brand-new key.
  if (state.key !== key) {
    return { messages: EMPTY_MESSAGES, loading: true, error: null };
  }

  return {
    messages: state.messages,
    loading: state.loading,
    error: state.error,
  };
}

/** Produce a compact display title for a session. */
export function sessionTitle(
  session: ChatSummary,
  firstUserMessage?: string,
): string {
  return deriveTitle(
    firstUserMessage || session.preview,
    i18n.t("chat.newChat"),
  );
}
