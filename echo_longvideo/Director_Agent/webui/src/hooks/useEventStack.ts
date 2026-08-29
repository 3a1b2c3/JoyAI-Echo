import { useState, useEffect, useCallback, useRef } from "react";

export interface EventSession {
  id: string;
  filename: string;
  event_count: number;
  size_bytes: number;
  modified: string;
  last_turn_id: string;
  last_session_key: string;
}

export interface EventRecord {
  seq: number;
  type: string;
  turn_id: string;
  timestamp: string;
  session_key: string;
  data: Record<string, any>;
}

export function useEventStack() {
  const [sessions, setSessions] = useState<EventSession[]>([]);
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const fetchIdRef = useRef(0);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch("/api/eventstack/sessions");
      if (res.ok) setSessions(await res.json());
    } catch (e) {
      console.error("Failed to fetch eventstack sessions", e);
    }
  }, []);

  const fetchEvents = useCallback(async (sessionId: string) => {
    const thisId = ++fetchIdRef.current;
    setLoading(true);
    try {
      const res = await fetch(`/api/eventstack/traces/${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();
      if (fetchIdRef.current === thisId) setEvents(data);
    } catch (e) {
      console.error("Failed to fetch events", e);
    } finally {
      if (fetchIdRef.current === thisId) setLoading(false);
    }
  }, []);

  const switchSession = useCallback((id: string | null) => {
    setActiveSession((prev) => {
      if (prev !== id) setEvents([]);
      return id;
    });
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  useEffect(() => {
    if (activeSession) {
      fetchEvents(activeSession);
    } else {
      fetchIdRef.current++;
      setEvents([]);
      setLoading(false);
    }
  }, [activeSession, fetchEvents]);

  return {
    sessions,
    activeSession,
    setActiveSession: switchSession,
    events,
    loading,
    refresh: useCallback(async () => {
      await fetchSessions();
      if (activeSession) fetchEvents(activeSession);
    }, [fetchSessions, fetchEvents, activeSession]),
  };
}
