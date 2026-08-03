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
  residual_update_norms?: number[][];
  mean_attentions?: number[][][];
  jspace?: JSpaceSummary;
  capture?: {
    window_tokens: number;
    truncated: boolean;
  };
}

export interface JSpaceToken {
  token_id: number;
  text: string;
  rank: number;
  logit: number;
}

export interface JSpaceSummary {
  method: "jacobian_lens";
  lens_prompts: number;
  lens_width: number;
  layers: number[];
  output_layer: number;
  tokens: string[];
  token_ids: number[];
  token_offset: number;
  truncated: boolean;
  top_tokens: JSpaceToken[][][];
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

export type SearchStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type SearchNodeKind = "brainstorm" | "executor" | "evaluator";

export interface SearchOptions {
  iterations: number;
  branches: number;
  branch_tokens: number;
  idea_tokens: number;
  evaluator_tokens: number;
  solved_threshold: number;
  seed: number;
}

export interface CreateSearchRequest {
  problem: string;
  goal: string;
  evaluator_rubric?: string | null;
  options: SearchOptions;
}

export interface SearchIdea {
  id: string;
  iteration: number;
  name: string;
  niche: string;
  executor_prompt: string;
  coordinates: Record<string, number>;
  parent_candidate_id: string | null;
  predicted_improvement: number | null;
}

export interface ResidualEvaluation {
  residuals: Record<string, number>;
  feedback: string;
  solved: boolean;
}

export interface CandidateRecord {
  id: string;
  idea: SearchIdea;
  run_id: string;
  evaluator_run_id: string;
  output: string;
  evaluation: ResidualEvaluation;
  score: number;
  created_at: string;
}

export interface SearchNode {
  id: string;
  kind: SearchNodeKind;
  iteration: number;
  label: string;
  status: RunStatus;
  idea_id: string | null;
  parent_node_id: string | null;
  run_id: string | null;
  output_excerpt: string | null;
  error: string | null;
}

export interface SearchRecord {
  id: string;
  status: SearchStatus;
  request: CreateSearchRequest;
  iteration: number;
  trust_radius: number;
  nodes: SearchNode[];
  candidates: CandidateRecord[];
  best_candidate_id: string | null;
  error: string | null;
  created_at: string;
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
  jspace_method: string | null;
  jspace_prompts: number | null;
  jspace_layers: number | null;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  model: ModelInfo;
  timestamp: string;
}
