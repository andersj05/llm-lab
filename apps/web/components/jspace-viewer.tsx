"use client";

import type { CSSProperties } from "react";
import { useMemo, useState } from "react";

import type { JSpaceSummary } from "@llm-lab/contracts";

interface JSpaceViewerProps {
  available: boolean;
  loading?: boolean;
  reason?: string | null;
  requested?: boolean;
  resultError?: string | null;
  summary?: JSpaceSummary;
}

type RankStyle = CSSProperties & { "--rank-heat": number };

export function JSpaceViewer({
  available,
  loading = false,
  reason,
  requested = false,
  resultError,
  summary,
}: JSpaceViewerProps) {
  const [selection, setSelection] = useState<{
    key: string;
    layer: number;
    position: number;
  } | null>(null);

  const frequentConcepts = useMemo(() => {
    if (!summary) return [];
    const scores = new Map<string, number>();
    const outputIndex = summary.layers.indexOf(summary.output_layer);
    const start = Math.floor(outputIndex / 3);
    const end = Math.max(start + 1, outputIndex - 2);
    for (let layer = start; layer < end; layer += 1) {
      for (const readout of summary.top_tokens[layer] ?? []) {
        const token = readout[0];
        if (!token) continue;
        scores.set(token.text, (scores.get(token.text) ?? 0) + 1 / (token.rank + 1));
      }
    }
    return [...scores.entries()]
      .toSorted((left, right) => right[1] - left[1])
      .slice(0, 8)
      .map(([text]) => text);
  }, [summary]);

  if (!summary) {
    return (
      <section className="panel jspace-panel">
        <div className="section-heading section-heading--compact">
          <div>
            <h2>J-space</h2>
            <p>Model-specific Jacobian-lens readout</p>
          </div>
          <span className={`availability-pill ${available ? "is-ready" : ""}`}>
            {loading && requested ? "Loading" : available ? "Lens ready" : "Unavailable"}
          </span>
        </div>
        <div className="jspace-empty">
          <strong>
            {loading && requested
              ? "Loading J-space readouts"
              : resultError && requested
                ? "J-space result could not be loaded"
                : requested
                  ? "No J-space data was stored"
                  : available
                    ? "J-space was not requested for this run"
                    : "A fitted lens is required"}
          </strong>
          <p>
            {loading && requested
              ? "The diagnostics artifact is being parsed. Large traced runs can take a moment."
              : resultError && requested
                ? resultError
                : requested
                  ? "Retry the diagnostics artifact or start a new run if capture failed."
                  : available
                    ? "Enable J-space before starting a new run to capture model-specific concepts."
                    : reason ?? "Fit a compatible Jacobian lens for this model."}
          </p>
          <a
            href="https://transformer-circuits.pub/2026/workspace/index.html"
            rel="noreferrer"
            target="_blank"
          >
            Read the method
          </a>
        </div>
      </section>
    );
  }

  const summaryKey = `${summary.token_offset}:${summary.tokens.length}:${summary.layers.join(",")}`;
  const outputIndex = summary.layers.indexOf(summary.output_layer);
  const selectedLayer =
    selection?.key === summaryKey
      ? selection.layer
      : Math.max(0, outputIndex > 1 ? outputIndex - 1 : outputIndex);
  const selectedPosition =
    selection?.key === summaryKey ? selection.position : Math.max(0, summary.tokens.length - 1);
  const selected = summary.top_tokens[selectedLayer]?.[selectedPosition] ?? [];
  const selectedLayerNumber = summary.layers[selectedLayer];
  const gridStyle = {
    gridTemplateColumns: `92px repeat(${summary.tokens.length}, minmax(76px, 1fr))`,
  };

  return (
    <section className="panel jspace-panel">
      <div className="section-heading section-heading--compact">
        <div>
          <h2>J-space</h2>
          <p>What each residual state is disposed to make the model verbalize later</p>
        </div>
        <span className="availability-pill is-ready">Jacobian lens</span>
      </div>

      <div className="jspace-summary">
        <div><span>Lens fit</span><strong>{summary.lens_prompts} prompts</strong></div>
        <div><span>Readout</span><strong>{summary.layers.length} layers</strong></div>
        <div><span>Window</span><strong>{summary.tokens.length} tokens</strong></div>
        <div><span>Frequent mid-layer concepts</span><strong>{frequentConcepts.join(" · ") || "—"}</strong></div>
      </div>

      <div className="jspace-explainer">
        <p>
          Each cell shows the highest-ranked word-like vocabulary token from the fitted average
          Jacobian. Darker cells rank closer to the top of Qwen’s full vocabulary. Select a cell
          to inspect its top five readouts.
        </p>
        <span>
          Experimental Qwen readout; this does not establish that the model has Anthropic’s
          reported Claude workspace or expose a literal reasoning transcript.
        </span>
      </div>

      <div className="jspace-layout">
        <div className="jspace-grid-scroll" tabIndex={0}>
          <div className="jspace-grid" style={gridStyle}>
            <div className="jspace-corner">Source token</div>
            {summary.tokens.map((token, position) => (
              <button
                className={`jspace-source-token ${selectedPosition === position ? "is-selected" : ""}`}
                key={`${token}-${position}`}
                onClick={() => setSelection({ key: summaryKey, layer: selectedLayer, position })}
                title={`Position ${summary.token_offset + position}: ${token}`}
                type="button"
              >
                {visibleToken(token)}
              </button>
            ))}
            {summary.layers.map((layer, layerIndex) => (
              <div className="jspace-row" key={layer} style={{ display: "contents" }}>
                <div className={`jspace-layer ${layer === summary.output_layer ? "is-output" : ""}`}>
                  <strong>L{layer.toString().padStart(2, "0")}</strong>
                  <span>{layerBand(layerIndex, summary.layers.length, layer === summary.output_layer)}</span>
                </div>
                {(summary.top_tokens[layerIndex] ?? []).map((readout, position) => {
                  const token = readout[0];
                  const heat = token ? Math.max(0.08, 1 - Math.log10(token.rank + 1) / 5.2) : 0;
                  return (
                    <button
                      className={`jspace-cell ${
                        selectedLayer === layerIndex && selectedPosition === position
                          ? "is-selected"
                          : ""
                      } ${layer === summary.output_layer ? "is-output" : ""}`}
                      key={`${layer}-${position}`}
                      onClick={() => {
                        setSelection({ key: summaryKey, layer: layerIndex, position });
                      }}
                      style={{ "--rank-heat": heat } as RankStyle}
                      title={token ? `${token.text}, full-vocabulary rank ${token.rank + 1}` : "No readout"}
                      type="button"
                    >
                      {token?.text || "—"}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <aside className="jspace-inspector">
          <span>Selected readout</span>
          <h3>
            Layer {selectedLayerNumber} · position {summary.token_offset + selectedPosition}
          </h3>
          <p>Source token: <code>{visibleToken(summary.tokens[selectedPosition] ?? "")}</code></p>
          <ol>
            {selected.map((token) => (
              <li key={token.token_id}>
                <strong>{token.text}</strong>
                <span>rank {token.rank + 1}</span>
              </li>
            ))}
          </ol>
        </aside>
      </div>
    </section>
  );
}

function visibleToken(token: string): string {
  return token.replaceAll(" ", "·").replaceAll("\n", "↵") || "∅";
}

function layerBand(index: number, count: number, output: boolean): string {
  if (output) return "output";
  const ratio = index / Math.max(1, count - 1);
  if (ratio < 0.33) return "early";
  if (ratio < 0.82) return "middle";
  return "late";
}
