import { describe, expect, it } from "vitest";

import { formatBytes, normalizeMatrix } from "./format";

describe("format helpers", () => {
  it("formats model weight sizes", () => {
    expect(formatBytes(1_503_300_328)).toBe("1.40 GB");
  });

  it("normalizes a trace matrix across layers", () => {
    expect(normalizeMatrix([[2, 4], [6, 10]])).toEqual([
      [0, 0.25],
      [0.5, 1],
    ]);
  });
});
