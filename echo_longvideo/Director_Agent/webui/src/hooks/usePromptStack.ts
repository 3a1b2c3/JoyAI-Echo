import { useState, useEffect, useCallback, useRef } from "react";

export interface PromptStackSession {
  id: string;
  filename: string;
  turn_count: number;
  size_bytes: number;
  modified: string;
  last_model: string;
  last_session_key: string;
}

export interface PromptPart {
  label: string;
  content: string;
  char_count: number;
}

export interface TraceMessage {
  role: string;
  preview: string;
  content: string;
  char_count: number;
  tool_calls?: { name: string; id: string; arguments?: string }[];
  tool_call_id?: string;
  tool_name?: string;
}

export interface TraceRecord {
  id: string;
  timestamp: string;
  session_key: string;
  iteration: number;
  model: string;
  parts: PromptPart[];
  messages_count: number;
  messages: TraceMessage[];
  response: {
    content?: string;
    tool_calls?: { name: string; arguments: string }[];
    reasoning_content?: string;
    usage?: Record<string, number>;
  };
}

export function usePromptStack() {
  const [sessions, setSessions] = useState<PromptStackSession[]>([]);
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [trace, setTrace] = useState<TraceRecord[]>([]);
  const [loading, setLoading] = useState(false);

  // Guard against stale fetch responses arriving after session switch
  const fetchIdRef = useRef(0);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch("/api/promptstack/sessions");
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
      }
    } catch (e) {
      console.error("Failed to fetch promptstack sessions", e);
    }
  }, []);

  const fetchTrace = useCallback(async (sessionId: string) => {
    // Bump fetch ID so any in-flight request for a previous session is ignored
    const thisId = ++fetchIdRef.current;
    setLoading(true);
    try {
      const res = await fetch(`/api/promptstack/traces/${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();
      // Only apply if this is still the most recent fetch
      if (fetchIdRef.current === thisId) {
        setTrace(data);
      }
    } catch (e) {
      console.error("Failed to fetch trace", e);
    } finally {
      if (fetchIdRef.current === thisId) {
        setLoading(false);
      }
    }
  }, []);

  // Wrap setActiveSession to immediately clear stale trace data
  const switchSession = useCallback(
    (id: string | null) => {
      setActiveSession((prev) => {
        if (prev !== id) {
          // Clear old data synchronously so UI never shows mismatched session/trace
          setTrace([]);
        }
        return id;
      });
    },
    [],
  );

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  useEffect(() => {
    if (activeSession) {
      fetchTrace(activeSession);
    } else {
      // No session selected — ensure trace is empty
      fetchIdRef.current++;
      setTrace([]);
      setLoading(false);
    }
  }, [activeSession, fetchTrace]);

  return {
    sessions,
    activeSession,
    setActiveSession: switchSession,
    trace,
    loading,
    refresh: useCallback(async () => {
      await fetchSessions();
      // Also re-fetch current trace to pick up new turns
      if (activeSession) {
        fetchTrace(activeSession);
      }
    }, [fetchSessions, fetchTrace, activeSession]),
  };
}
