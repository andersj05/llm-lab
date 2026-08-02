"use client";

import type { CSSProperties } from "react";

import { normalizeMatrix } from "@/lib/format";

interface TraceMatrixProps {
  tokens?: string[];
  values?: number[][];
}

type HeatStyle = CSSProperties & { "--heat": number };

export function TraceMatrix({ tokens, values }: TraceMatrixProps) {
  if (!tokens?.length || !values?.length) {
    return (
      <div className="empty-trace">
        <span className="empty-trace__mark">∅</span>
        <div>
          <strong>No activation trace captured</strong>
          <p>Enable hidden states for a run to populate the layer × token field.</p>
        </div>
      </div>
    );
  }

  const normalized = normalizeMatrix(values);
  const gridStyle = {
    gridTemplateColumns: `76px repeat(${tokens.length}, minmax(28px, 1fr))`,
  };

  return (
    <div className="matrix-scroll" tabIndex={0} aria-label="Layer by token activation matrix">
      <div className="trace-matrix" style={gridStyle}>
        <div className="matrix-corner">TOKEN</div>
        {tokens.map((token, index) => (
          <div className="matrix-token" key={`${token}-${index}`} title={token}>
            {token.replace("Ġ", "·").slice(0, 8)}
          </div>
        ))}
        {values.map((row, layer) => (
          <div className="matrix-row" key={`layer-${layer}`} style={{ display: "contents" }}>
            <div className="matrix-layer">L{layer.toString().padStart(2, "0")}</div>
            {row.map((value, token) => (
              <div
                className="matrix-cell"
                key={`${layer}-${token}`}
                style={{ "--heat": normalized[layer]?.[token] ?? 0 } as HeatStyle}
                title={`Layer ${layer}, ${tokens[token]}: ${value.toFixed(4)}`}
                aria-label={`Layer ${layer}, token ${tokens[token]}, norm ${value.toFixed(4)}`}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
