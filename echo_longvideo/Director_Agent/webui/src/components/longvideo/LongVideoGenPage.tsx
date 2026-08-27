import { ThreadWorkplaceShell } from "@/components/director/ThreadWorkplaceShell";
import {
  WorkflowSelector,
  type WorkflowMode,
} from "@/components/director/WorkflowSelector";
import { WorkplaceProvider } from "@/providers/WorkplaceProvider";
import type { ChatSummary } from "@/lib/types";

interface LongVideoGenPageProps {
  mode: "unselected" | WorkflowMode;
  onModeChange: (mode: WorkflowMode) => void;
  session: ChatSummary | null;
  title: string;
  onToggleSidebar: () => void;
  onGoHome: () => void;
  onNewChat: () => Promise<string | null>;
  hideSidebarToggleOnDesktop?: boolean;
  onReplyEnd?: () => void;
}

export function LongVideoGenPage(props: LongVideoGenPageProps) {
  if (props.mode === "unselected") {
    return <WorkflowSelector onSelect={props.onModeChange} />;
  }
  const sessionKey = props.session?.key ?? null;
  const { mode, onModeChange: _onModeChange, ...shellProps } = props;
  return (
    <WorkplaceProvider sessionKey={sessionKey}>
      <ThreadWorkplaceShell {...shellProps} quickMode={mode === "quick"} />
    </WorkplaceProvider>
  );
}
