import { useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useClient } from "@/providers/ClientProvider";
import { listPeSets, type PeSet } from "@/lib/pe-api";

/** Per-session Prompt Engineering set selector (internal test page). */
export function PeSelector({ chatId }: { chatId?: string }) {
  const { client, token } = useClient();
  const [sets, setSets] = useState<PeSet[]>([]);
  const [active, setActive] = useState<string>("");
  const [enabled, setEnabled] = useState(true);

  useEffect(() => {
    let cancelled = false;
    listPeSets(token)
      .then((res) => {
        if (cancelled) return;
        setSets(res.sets);
        setActive(res.active);
        setEnabled(res.enabled);
      })
      .catch(() => {
        if (!cancelled) setEnabled(false);
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    return client.onPe((evChatId, next) => {
      // Only reflect changes for the chat this selector is bound to.
      if (chatId && evChatId === chatId) setActive(next);
    });
  }, [client, chatId]);

  if (!enabled || sets.length === 0) return null;

  const selected = sets.find((s) => s.name === active);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          disabled={!chatId}
          aria-label="Switch prompt kit"
          className="h-7 gap-1.5 rounded-md px-2 text-[11px] text-muted-foreground hover:bg-accent/35 hover:text-foreground"
        >
          <FlaskConical className="h-3.5 w-3.5" />
          <span className="max-w-[7rem] truncate">
            {selected?.label ?? active}
          </span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>Prompt Kit</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          value={active}
          onValueChange={(value) => {
            if (value === active || !chatId) return;
            setActive(value);
            client.setPe(value, chatId);
          }}
        >
          {sets.map((option) => (
            <DropdownMenuRadioItem key={option.name} value={option.name}>
              <span className="flex min-w-0 flex-col gap-0.5">
                <span>{option.label}</span>
                {option.description ? (
                  <span className="truncate text-xs text-muted-foreground">
                    {option.description}
                  </span>
                ) : null}
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
