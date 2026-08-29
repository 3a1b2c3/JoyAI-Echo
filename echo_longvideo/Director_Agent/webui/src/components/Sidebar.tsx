import { ChevronLeft, PanelLeftClose, Plus } from "lucide-react";
import { useTranslation } from "react-i18next";

import { ChatList } from "@/components/ChatList";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { ChatSummary } from "@/lib/types";

interface SidebarProps {
  sessions: ChatSummary[];
  activeKey: string | null;
  loading: boolean;
  onNewChat: () => void;
  onSelect: (key: string) => void;
  onRequestDelete: (key: string, label: string) => void;
  onCollapse: () => void;
}

export function Sidebar(props: SidebarProps) {
  const { t } = useTranslation();
  const goBack = () => {
    window.history.back();
  };
  return (
    <aside className="flex h-full w-full flex-col border-r border-sidebar-border/70 bg-sidebar text-sidebar-foreground">
      <div className="flex items-center justify-between px-2 py-2">
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("sidebar.collapse")}
          onClick={goBack}
          className="h-7 w-7 rounded-lg text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground"
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </Button>
      </div>
      <div className="flex items-center justify-between px-2 py-2">
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("sidebar.collapse")}
          onClick={props.onCollapse}
          className="h-7 w-7 rounded-lg text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground"
        >
          <PanelLeftClose className="h-3.5 w-3.5" />
        </Button>
      </div>
      <div className="px-2 pb-2.5">
        <Button
          onClick={props.onNewChat}
          className="h-8.5 w-full justify-start gap-2 rounded-lg border border-sidebar-border/80 bg-card/25 px-3 text-[13px] font-medium text-sidebar-foreground shadow-none hover:bg-sidebar-accent/80"
          variant="outline"
        >
          <Plus className="h-3.5 w-3.5" />
          {t("sidebar.newChat")}
        </Button>
      </div>
      <Separator className="bg-sidebar-border/70" />
      <div className="flex items-center justify-between px-2.5 py-2 text-[11px] font-medium text-muted-foreground">
        <span>{t("sidebar.recent")}</span>
      </div>
      <div className="flex-1 overflow-hidden">
        <ChatList
          sessions={props.sessions}
          activeKey={props.activeKey}
          loading={props.loading}
          onSelect={props.onSelect}
          onRequestDelete={props.onRequestDelete}
        />
      </div>
    </aside>
  );
}
