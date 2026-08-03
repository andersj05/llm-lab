export function formatBytes(value: number | null): string {
  if (value === null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit > 1 ? 2 : 0)} ${units[unit]}`;
}

export function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

export function normalizeMatrix(values: number[][]): number[][] {
  return values.map((row) => {
    const sorted = row.filter(Number.isFinite).toSorted((left, right) => left - right);
    if (!sorted.length) return row.map(() => 0.5);
    const low = sorted[0];
    const high = sorted[Math.min(sorted.length - 1, Math.max(1, Math.floor((sorted.length - 1) * 0.95)))];
    if (low === high) return row.map(() => 0.5);
    return row.map((value) => Math.max(0, Math.min(1, (value - low) / (high - low))));
  });
}
