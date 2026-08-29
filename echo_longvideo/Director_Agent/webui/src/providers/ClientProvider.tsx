import { createContext, useContext, type ReactNode } from "react";

import type { NanobotClient } from "@/lib/nanobot-client";

interface ClientContextValue {
  client: NanobotClient;
  token: string;
  modelName: string | null;
  webuiUserId: string | null;
}

const ClientContext = createContext<ClientContextValue | null>(null);

export function ClientProvider({
  client,
  token,
  modelName = null,
  webuiUserId = null,
  children,
}: {
  client: NanobotClient;
  token: string;
  modelName?: string | null;
  webuiUserId?: string | null;
  children: ReactNode;
}) {
  return (
    <ClientContext.Provider value={{ client, token, modelName, webuiUserId }}>
      {children}
    </ClientContext.Provider>
  );
}

export function useClient(): ClientContextValue {
  const ctx = useContext(ClientContext);
  if (!ctx) {
    throw new Error("useClient must be used within a ClientProvider");
  }
  return ctx;
}
