import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Clapperboard, PanelLeftOpen } from "lucide-react";
import { DeleteConfirm } from "@/components/DeleteConfirm";
import { LongVideoGenPage } from "@/components/longvideo/LongVideoGenPage";
import type { WorkflowMode } from "@/components/director/WorkflowSelector";
import { Sidebar } from "@/components/Sidebar";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { preloadMarkdownText } from "@/components/MarkdownText";
import { useSessions } from "@/hooks/useSessions";
import { cn } from "@/lib/utils";
import { deriveWsUrl, fetchBootstrap } from "@/lib/bootstrap";
import { deriveTitle } from "@/lib/format";
import { NanobotClient } from "@/lib/nanobot-client";
import { webuiSessionKey, webuiChatIdFromKey } from "@/lib/session-key";
import { ClientProvider, useClient } from "@/providers/ClientProvider";
import type { ChatSummary } from "@/lib/types";

type BootState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready";
      client: NanobotClient;
      token: string;
      modelName: string | null;
      webuiUserId: string | null;
    };

const SIDEBAR_STORAGE_KEY = "nanobot-webui.sidebar";
const SIDEBAR_WIDTH = 279;
const AUTH_REFRESH_SKEW_S = 30;
const AUTH_REFRESH_RETRY_MS = 10_000;

function readSidebarOpen(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const raw = window.localStorage.getItem(SIDEBAR_STORAGE_KEY);
    if (raw === null) return true;
    return raw === "1";
  } catch {
    return true;
  }
}

export default function App() {
  const { t } = useTranslation();
  const [state, setState] = useState<BootState>({ status: "loading" });
  const [bootAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let refreshTimer: ReturnType<typeof setTimeout> | null = null;
    let clientInstance: NanobotClient | null = null;

    const clearRefreshTimer = () => {
      if (refreshTimer !== null) {
        clearTimeout(refreshTimer);
        refreshTimer = null;
      }
    };

    const scheduleRefresh = (client: NanobotClient, expiresIn: number) => {
      clearRefreshTimer();
      const delayMs = Math.max(
        5_000,
        (expiresIn - AUTH_REFRESH_SKEW_S) * 1_000,
      );
      refreshTimer = setTimeout(() => {
        void refreshAuth(client);
      }, delayMs);
    };

    const refreshAuth = async (
      client: NanobotClient,
    ): Promise<string | null> => {
      try {
        const refreshed = await fetchBootstrap();
        const url = deriveWsUrl(refreshed.ws_path, refreshed.token);
        client.updateUrl(url);
        if (!cancelled) {
          setState((current) =>
            current.status === "ready" && current.client === client
              ? {
                  ...current,
                  token: refreshed.token,
                  modelName: refreshed.model_name ?? current.modelName,
                  webuiUserId: refreshed.user_id ?? current.webuiUserId,
                }
              : current,
          );
          scheduleRefresh(client, refreshed.expires_in);
        }
        return url;
      } catch (e) {
        if (!cancelled) {
          clearRefreshTimer();
          refreshTimer = setTimeout(() => {
            void refreshAuth(client);
          }, AUTH_REFRESH_RETRY_MS);
        }
        return null;
      }
    };

    (async () => {
      try {
        const boot = await fetchBootstrap();
        if (cancelled) return;
        const url = deriveWsUrl(boot.ws_path, boot.token);
        let client: NanobotClient;
        client = new NanobotClient({
          url,
          onReauth: async () => {
            return refreshAuth(client);
          },
        });
        clientInstance = client;
        client.connect();
        setState({
          status: "ready",
          client,
          token: boot.token,
          modelName: boot.model_name ?? null,
          webuiUserId: boot.user_id ?? null,
        });
        scheduleRefresh(client, boot.expires_in);
      } catch (e) {
        if (cancelled) return;
        setState({ status: "error", message: (e as Error).message });
      }
    })();
    return () => {
      cancelled = true;
      clearRefreshTimer();
      clientInstance?.close();
    };
  }, [bootAttempt]);

  useEffect(() => {
    const warm = () => preloadMarkdownText();
    const win = globalThis as typeof globalThis & {
      requestIdleCallback?: (
        callback: IdleRequestCallback,
        options?: IdleRequestOptions,
      ) => number;
      cancelIdleCallback?: (handle: number) => void;
    };
    if (typeof win.requestIdleCallback === "function") {
      const id = win.requestIdleCallback(warm, { timeout: 1500 });
      return () => win.cancelIdleCallback?.(id);
    }
    const id = globalThis.setTimeout(warm, 250);
    return () => globalThis.clearTimeout(id);
  }, []);

  if (state.status === "loading") {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <div className="flex flex-col items-center gap-3 animate-in fade-in-0 duration-300">
          <Clapperboard
            className="h-10 w-10 animate-pulse text-foreground/60"
            aria-hidden
          />
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-foreground/40" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-foreground/60" />
            </span>
            {t("app.loading.connecting")}
          </div>
        </div>
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="flex h-full w-full items-center justify-center px-4 text-center">
        <div className="flex max-w-md flex-col items-center gap-3">
          <Clapperboard
            className="h-10 w-10 text-foreground/40"
            aria-hidden
          />
          <p className="text-lg font-semibold">{t("app.error.title")}</p>
          <p className="text-sm text-muted-foreground">{state.message}</p>
          <p className="text-xs text-muted-foreground">
            {t("app.error.gatewayHint")}
          </p>
        </div>
      </div>
    );
  }

  return (
    <ClientProvider
      client={state.client}
      token={state.token}
      modelName={state.modelName}
      webuiUserId={state.webuiUserId}
    >
      <Shell />
    </ClientProvider>
  );
}

function Shell() {
  const { t, i18n } = useTranslation();
  const { webuiUserId } = useClient();
  const { sessions, loading, refresh, createChat, deleteChat } = useSessions();
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [desktopSidebarOpen, setDesktopSidebarOpen] =
    useState<boolean>(readSidebarOpen);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<{
    key: string;
    label: string;
  } | null>(null);
  const lastSessionsLen = useRef(0);
  const skipAutoSelectRef = useRef(false);
  const pendingNewChatRef = useRef(false);
  const creatingChatRef = useRef(false);
  const [workflowMode, setWorkflowMode] = useState<
    "unselected" | WorkflowMode
  >("unselected");

  useEffect(() => {
    if (!activeKey) {
      if (!pendingNewChatRef.current) setWorkflowMode("unselected");
      return;
    }
    pendingNewChatRef.current = false;
    const session = sessions.find((item) => item.key === activeKey);
    setWorkflowMode(session?.autoGenerate ? "quick" : "director");
  }, [activeKey, sessions]);
  useEffect(() => {
    try {
      window.localStorage.setItem(
        SIDEBAR_STORAGE_KEY,
        desktopSidebarOpen ? "1" : "0",
      );
    } catch {
      // ignore storage errors (private mode, etc.)
    }
  }, [desktopSidebarOpen]);

  useEffect(() => {
    if (activeKey) return;
    if (skipAutoSelectRef.current) {
      skipAutoSelectRef.current = false;
      lastSessionsLen.current = sessions.length;
      return;
    }
    if (sessions.length > 0 && lastSessionsLen.current === 0) {
      setActiveKey(sessions[0].key);
    }
    lastSessionsLen.current = sessions.length;
  }, [sessions, activeKey]);

  useEffect(() => {
    if (!activeKey || !webuiUserId) return;
    const chatId = webuiChatIdFromKey(activeKey);
    const canonical = webuiSessionKey(webuiUserId, chatId);
    if (canonical === activeKey) return;
    if (sessions.some((s) => s.key === canonical)) {
      setActiveKey(canonical);
    }
  }, [activeKey, sessions, webuiUserId]);

  const activeSession = useMemo<ChatSummary | null>(() => {
    if (!activeKey) return null;
    return sessions.find((s) => s.key === activeKey) ?? null;
  }, [sessions, activeKey]);

  const closeDesktopSidebar = useCallback(() => {
    setDesktopSidebarOpen(false);
  }, []);

  const openDesktopSidebar = useCallback(() => {
    setDesktopSidebarOpen(true);
  }, []);

  const closeMobileSidebar = useCallback(() => {
    setMobileSidebarOpen(false);
  }, []);

  const toggleSidebar = useCallback(() => {
    const isDesktop =
      typeof window !== "undefined" &&
      window.matchMedia("(min-width: 1024px)").matches;
    if (isDesktop) {
      setDesktopSidebarOpen((v) => !v);
    } else {
      setMobileSidebarOpen((v) => !v);
    }
  }, []);

  const onNewChat = useCallback(async () => {
    try {
      const chatId = await createChat({
        autoGenerate: workflowMode === "quick",
      });
      pendingNewChatRef.current = false;
      setActiveKey(webuiSessionKey(webuiUserId, chatId));
      setMobileSidebarOpen(false);
      return chatId;
    } catch (e) {
      console.error("Failed to create chat", e);
      return null;
    }
  }, [createChat, webuiUserId, workflowMode]);

  const onSidebarNewChat = useCallback(() => {
    pendingNewChatRef.current = true;
    skipAutoSelectRef.current = true;
    setActiveKey(null);
    setWorkflowMode("unselected");
    setMobileSidebarOpen(false);
  }, []);

  const onWorkflowSelect = useCallback(
    (mode: WorkflowMode) => {
      if (activeKey || creatingChatRef.current) return;
      creatingChatRef.current = true;
      pendingNewChatRef.current = true;
      setWorkflowMode(mode);
      void createChat({ autoGenerate: mode === "quick" })
        .then((chatId) => {
          setActiveKey(webuiSessionKey(webuiUserId, chatId));
        })
        .catch((error) => {
          console.error("Failed to create chat", error);
          setWorkflowMode("unselected");
        })
        .finally(() => {
          pendingNewChatRef.current = false;
          creatingChatRef.current = false;
        });
    },
    [activeKey, createChat, webuiUserId],
  );

  const onSelectChat = useCallback((key: string) => {
    pendingNewChatRef.current = false;
    setActiveKey(key);
    setMobileSidebarOpen(false);
  }, []);

  const onConfirmDelete = useCallback(async () => {
    if (!pendingDelete) return;
    const key = pendingDelete.key;
    const deletingActive = activeKey === key;
    const currentIndex = sessions.findIndex((s) => s.key === key);
    const fallbackKey = deletingActive
      ? (sessions[currentIndex + 1]?.key ??
        sessions[currentIndex - 1]?.key ??
        null)
      : activeKey;
    setPendingDelete(null);
    if (deletingActive) setActiveKey(fallbackKey);
    if (pendingNewChatRef.current) {
      pendingNewChatRef.current = false;
    }
    try {
      await deleteChat(key);
    } catch (e) {
      if (deletingActive) setActiveKey(key);
      console.error("Failed to delete session", e);
    }
  }, [pendingDelete, deleteChat, activeKey, sessions]);

  const headerTitle = activeSession
    ? deriveTitle(
        activeSession.preview,
        t("chat.fallbackTitle", { id: activeSession.chatId.slice(0, 6) }),
      )
    : t("app.brand");

  useEffect(() => {
    document.title = activeSession
      ? t("app.documentTitle.chat", { title: headerTitle })
      : t("app.documentTitle.base");
  }, [activeSession, headerTitle, i18n.resolvedLanguage, t]);

  const sidebarProps = {
    sessions,
    activeKey,
    loading,
    onNewChat: () => {
      void onSidebarNewChat();
    },
    onSelect: onSelectChat,
    onRequestDelete: (key: string, label: string) =>
      setPendingDelete({ key, label }),
  };

  return (
    <div className="relative flex h-full w-full overflow-hidden">
      {/* Desktop sidebar: in normal flow, so the thread area width stays honest. */}
      <aside
        className={cn(
          "relative z-20 hidden shrink-0 overflow-hidden lg:block",
          "transition-[width] duration-300 ease-out",
        )}
        style={{ width: desktopSidebarOpen ? SIDEBAR_WIDTH : 0 }}
      >
        <div
          className={cn(
            "absolute inset-y-0 left-0 h-full w-[279px] overflow-hidden bg-sidebar shadow-inner-right",
            "transition-transform duration-300 ease-out",
            desktopSidebarOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <Sidebar {...sidebarProps} onCollapse={closeDesktopSidebar} />
        </div>
      </aside>

      {/* 桌面端侧栏收起后，由布局层提供展开入口（不依赖各业务页 Header） */}
      {!desktopSidebarOpen ? (
        <div className="absolute left-0 top-0 z-30 hidden px-3 py-2 lg:block">
          <Button
            variant="ghost"
            size="icon"
            aria-label={t("thread.header.toggleSidebar")}
            onClick={openDesktopSidebar}
            className="h-7 w-7 rounded-md text-muted-foreground hover:bg-accent/35 hover:text-foreground"
          >
            <PanelLeftOpen className="h-3.5 w-3.5" />
          </Button>
        </div>
      ) : null}

      <Sheet
        open={mobileSidebarOpen}
        onOpenChange={(open) => setMobileSidebarOpen(open)}
      >
        <SheetContent
          side="left"
          showCloseButton={false}
          className="w-[279px] p-0 sm:max-w-[279px] lg:hidden"
        >
          <Sidebar {...sidebarProps} onCollapse={closeMobileSidebar} />
        </SheetContent>
      </Sheet>

      <main className="flex h-full min-w-0 flex-1 flex-col">
        <LongVideoGenPage
          mode={workflowMode}
          onModeChange={onWorkflowSelect}
          session={activeSession}
          title={headerTitle}
          onToggleSidebar={toggleSidebar}
          onGoHome={() => setActiveKey(null)}
          onNewChat={onNewChat}
          hideSidebarToggleOnDesktop
          onReplyEnd={() => {
            // Preview is written after the turn saves; brief delay matches manual refresh timing.
            window.setTimeout(() => {
              void refresh();
            }, 1500);
          }}
        />
      </main>

      <DeleteConfirm
        open={!!pendingDelete}
        title={pendingDelete?.label ?? ""}
        onCancel={() => setPendingDelete(null)}
        onConfirm={onConfirmDelete}
      />
    </div>
  );
}
