export type RunStatus =
  | "queued"
  | "loading"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type EventType =
  | "run.queued"
  | "run.started"
  | "model.loading"
  | "model.ready"
  | "token.generated"
  | "trace.ready"
  | "run.completed"
  | "run.failed"
  | "run.cancelled";

export interface GenerationOptions {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  do_sample: boolean;
  seed: number;
  enable_thinking: boolean;
}

export interface TraceOptions {
  logprobs: boolean;
  hidden_states: boolean;
  attentions: boolean;
  jspace: boolean;
  persist_raw: boolean;
}

export interface CreateRunRequest {
  prompt: string;
  system_prompt?: string | null;
  generation: GenerationOptions;
  traces: TraceOptions;
}

export interface TokenAlternative {
  token_id: number;
  text: string;
  logprob: number;
}

export interface TokenTrace {
  index: number;
  token_id: number;
  text: string;
  logprob: number | null;
  top_alternatives: TokenAlternative[];
}

export interface TraceSummary {
  tokens?: string[];
  hidden_state_norms?: number[][];
  mean_attentions?: number[][][];
  capture?: {
    window_tokens: number;
    truncated: boolean;
  };
}

export interface GenerationSummary {
  input_tokens: number;
  generated_tokens: number;
  requested_max_new_tokens: number;
  effective_max_new_tokens: number;
  context_limit: number;
  logprobs_captured: boolean;
  logprob_token_limit: number;
}

export interface RunResult {
  output_text: string;
  tokens: TokenTrace[];
  generation?: GenerationSummary;
  traces: TraceSummary;
}

export interface DeleteRunResponse {
  id: string;
  deleted: boolean;
}

export interface RunRecord {
  id: string;
  status: RunStatus;
  prompt: string;
  system_prompt: string | null;
  request: CreateRunRequest;
  output_text: string | null;
  error: string | null;
  tokens_generated: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface RunEvent {
  run_id: string;
  sequence: number;
  type: EventType;
  created_at: string;
  payload: Record<string, unknown>;
}

export interface ModelInfo {
  path: string;
  available: boolean;
  loaded: boolean;
  architecture: string | null;
  parameter_dtype: string | null;
  layers: number | null;
  hidden_size: number | null;
  vocabulary_size: number | null;
  max_context_tokens: number | null;
  weight_bytes: number | null;
  device: string | null;
  jspace_available: boolean;
  jspace_reason: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  model: ModelInfo;
  timestamp: string;
}
