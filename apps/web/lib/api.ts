import type {
  CreateRunRequest,
  CreateSearchRequest,
  DeleteRunResponse,
  HealthResponse,
  RunEvent,
  RunRecord,
  RunResult,
  SearchRecord,
} from "@llm-lab/contracts";

const API_URL = (
  process.env.NEXT_PUBLIC_LLM_LAB_API_URL ?? "http://localhost:8000/api/v1"
).replace(/\/$/, "");

const WS_URL = (
  process.env.NEXT_PUBLIC_LLM_LAB_WS_URL ?? "ws://localhost:8000/api/v1"
).replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  runs: () => request<RunRecord[]>("/runs"),
  run: (runId: string) => request<RunRecord>(`/runs/${runId}`),
  events: (runId: string) => request<RunEvent[]>(`/runs/${runId}/events`),
  searches: () => request<SearchRecord[]>("/searches"),
  search: (searchId: string) => request<SearchRecord>(`/searches/${searchId}`),
  createSearch: (payload: CreateSearchRequest) =>
    request<SearchRecord>("/searches", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  cancelSearch: (searchId: string) =>
    request<SearchRecord>(`/searches/${searchId}/cancel`, {
      method: "POST",
    }),
  createRun: (payload: CreateRunRequest) =>
    request<RunRecord>("/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  cancelRun: (runId: string) =>
    request<RunRecord>(`/runs/${runId}/cancel`, { method: "POST" }),
  deleteRun: (runId: string) =>
    request<DeleteRunResponse>(`/runs/${runId}`, { method: "DELETE" }),
  result: (runId: string) =>
    request<RunResult>(`/runs/${runId}/result`),
};

export function subscribeToRun(
  runId: string,
  after: number,
  onEvent: (event: RunEvent) => void,
  onError: () => void,
): () => void {
  const socket = new WebSocket(`${WS_URL}/runs/${runId}/events/ws?after=${after}`);
  socket.addEventListener("message", (message) => {
    onEvent(JSON.parse(message.data) as RunEvent);
  });
  socket.addEventListener("error", onError);
  return () => socket.close();
}
