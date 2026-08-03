import { describe, expect, it } from "vitest";

import { formatBytes, normalizeMatrix } from "./format";

describe("format helpers", () => {
  it("formats model weight sizes", () => {
    expect(formatBytes(1_503_300_328)).toBe("1.40 GB");
  });

  it("normalizes each layer independently", () => {
    expect(normalizeMatrix([[2, 4], [6, 10]])).toEqual([
      [0, 1],
      [0, 1],
    ]);
  });

  it("clips a single attention-sink outlier", () => {
    const normalized = normalizeMatrix([[1, 2, 3, 4, 100]]);
    expect(normalized[0]).toEqual([0, 1 / 3, 2 / 3, 1, 1]);
  });
});
