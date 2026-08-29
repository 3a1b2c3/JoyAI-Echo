import { ApiError } from "./api";

export interface PeSet {
  name: string;
  label: string;
  description: string;
}

export interface PeListResponse {
  ok: boolean;
  sets: PeSet[];
  active: string;
  enabled: boolean;
}

/** Fetch the available Prompt Engineering sets and the currently active one. */
export async function listPeSets(
  token: string,
  base: string = "",
): Promise<PeListResponse> {
  const res = await fetch(`${base}/api/pe-sets`, {
    headers: { Authorization: `Bearer ${token}` },
    credentials: "same-origin",
  });
  if (!res.ok) {
    const detail = (await res.text()).trim();
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as PeListResponse;
}
