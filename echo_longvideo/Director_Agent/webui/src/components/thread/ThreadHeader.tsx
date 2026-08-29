import { PanelLeftOpen } from "lucide-react";
import { useTranslation } from "react-i18next";

import { PeSelector } from "@/components/PeSelector";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ThreadHeaderProps {
  title: string;
  onToggleSidebar: () => void;
  onGoHome: () => void;
  hideSidebarToggleOnDesktop?: boolean;
  chatId?: string;
}

export function ThreadHeader({
  title,
  onToggleSidebar,
  onGoHome,
  hideSidebarToggleOnDesktop = false,
  chatId,
}: ThreadHeaderProps) {
  const { t } = useTranslation();
  return (
    <div className="relative z-10 flex items-center justify-between gap-3 px-3 py-2">
      <div className="relative flex min-w-0 items-center gap-2">
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("thread.header.toggleSidebar")}
          onClick={onToggleSidebar}
          className={cn(
            "h-7 w-7 rounded-md text-muted-foreground hover:bg-accent/35 hover:text-foreground",
            hideSidebarToggleOnDesktop && "lg:pointer-events-none lg:opacity-0",
          )}
        >
          <PanelLeftOpen className="h-3.5 w-3.5" />
        </Button>
        <button
          type="button"
          onClick={onGoHome}
          className="flex min-w-0 items-center gap-2 rounded-md px-1.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:bg-accent/35 hover:text-foreground"
        >
          <span className="max-w-[min(60vw,32rem)] truncate">{title}</span>
        </button>
      </div>

      <PeSelector chatId={chatId} />

      <div aria-hidden className="pointer-events-none absolute inset-x-0 top-full h-4" />
    </div>
  );
}
