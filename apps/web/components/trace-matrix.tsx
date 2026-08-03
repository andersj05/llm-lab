"use client";

import type { CSSProperties } from "react";
import { useState } from "react";

import { normalizeMatrix } from "@/lib/format";

interface TraceMatrixProps {
  tokens?: string[];
  residualUpdates?: number[][];
  residualNorms?: number[][];
  emptyTitle?: string;
  emptyMessage?: string;
}

type HeatStyle = CSSProperties & { "--heat": number };

export function TraceMatrix({
  tokens,
  residualUpdates,
  residualNorms,
  emptyTitle = "No activation data",
  emptyMessage = "Enable Activations before starting the run.",
}: TraceMatrixProps) {
  const [mode, setMode] = useState<"updates" | "norms">("updates");
  const values = mode === "updates" ? residualUpdates : residualNorms;

  if (!tokens?.length || !values?.length) {
    return (
      <div className="empty-trace">
        <span className="empty-trace__mark">—</span>
        <div>
          <strong>{emptyTitle}</strong>
          <p>{emptyMessage}</p>
        </div>
      </div>
    );
  }

  const normalized = normalizeMatrix(values);
  const gridStyle = {
    gridTemplateColumns: `76px repeat(${tokens.length}, minmax(28px, 1fr))`,
  };

  return (
    <>
      <div className="trace-toolbar">
        <div className="segmented-control" aria-label="Activation metric">
          <button
            className={mode === "updates" ? "is-active" : ""}
            onClick={() => setMode("updates")}
            type="button"
          >
            Layer contribution
          </button>
          <button
            className={mode === "norms" ? "is-active" : ""}
            onClick={() => setMode("norms")}
            type="button"
          >
            Residual magnitude
          </button>
        </div>
        <p>
          {mode === "updates"
            ? "L2 magnitude of what each transformer layer writes to each token."
            : "L2 magnitude of the accumulated residual stream after each layer."}
          {" "}Color is scaled within each layer and clips the top 5% outlier.
        </p>
      </div>
      <div className="matrix-scroll" tabIndex={0} aria-label="Layer by token activation matrix">
        <div className="trace-matrix" style={gridStyle}>
          <div className="matrix-corner">Token</div>
          {tokens.map((token, index) => (
            <div className="matrix-token" key={`${token}-${index}`} title={token}>
              {token.replace("Ġ", "·").slice(0, 8)}
            </div>
          ))}
          {values.map((row, layer) => {
            const label = mode === "norms" && layer === 0 ? "Embed" : `L${(
              mode === "norms" ? layer - 1 : layer
            )
              .toString()
              .padStart(2, "0")}`;
            return (
              <div className="matrix-row" key={`layer-${layer}`} style={{ display: "contents" }}>
                <div className="matrix-layer">{label}</div>
                {row.map((value, token) => (
                  <div
                    className="matrix-cell"
                    key={`${layer}-${token}`}
                    style={{ "--heat": normalized[layer]?.[token] ?? 0 } as HeatStyle}
                    title={`${label}, ${tokens[token]}: ${value.toFixed(4)}`}
                    aria-label={`${label}, token ${tokens[token]}, value ${value.toFixed(4)}`}
                  />
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}
