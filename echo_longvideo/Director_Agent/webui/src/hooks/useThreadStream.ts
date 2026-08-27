import { useEffect, useMemo, useRef } from "react";

import { useNanobotStream } from "@/hooks/useNanobotStream";
import { useSessionHistory } from "@/hooks/useSessions";
import type { ChatSummary, UIMessage } from "@/lib/types";

export type ThreadStreamControl = ReturnType<typeof useThreadStream>;

export function useThreadStream(
  session: ChatSummary | null,
  options?: { onReplyEnd?: () => void },
) {
  const chatId = session?.chatId ?? null;
  const historyKey = session?.key ?? null;
  const { messages: historical, loading } = useSessionHistory(historyKey);
  const messageCacheRef = useRef<Map<string, UIMessage[]>>(new Map());
  /** Skip one cache write after chatId changes (messages may still be stale). */
  const skipCacheWriteRef = useRef(false);

  const initial = useMemo(() => {
    if (!chatId) return historical;
    const cached = messageCacheRef.current.get(chatId);
    return cached?.length ? cached : historical;
  }, [chatId, historical]);

  const {
    messages,
    isStreaming,
    turnComplete,
    send,
    answerQuestion,
    setMessages,
    streamError,
    dismissStreamError,
  } = useNanobotStream(chatId, initial, {
    onReplyEnd: options?.onReplyEnd,
  });

  useEffect(() => {
    if (!chatId || loading) return;
    const cached = messageCacheRef.current.get(chatId);
    // When the user switches away and back, keep the local in-memory thread
    // state (including not-yet-persisted messages) instead of replacing it with
    // whatever the history endpoint currently knows about.
    setMessages(cached && cached.length > 0 ? cached : historical);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, chatId, historical]);

  useEffect(() => {
    if (chatId) return;
    setMessages(historical);
  }, [chatId, historical, setMessages]);

  useEffect(() => {
    skipCacheWriteRef.current = true;
  }, [chatId]);

  useEffect(() => {
    if (!chatId) return;
    if (skipCacheWriteRef.current) {
      skipCacheWriteRef.current = false;
      return;
    }
    messageCacheRef.current.set(chatId, messages);
  }, [chatId, messages]);

  return {
    chatId,
    loading,
    messages,
    isStreaming,
    turnComplete,
    send,
    answerQuestion,
    setMessages,
    streamError,
    dismissStreamError,
  };
}
