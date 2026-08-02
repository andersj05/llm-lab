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
  const flat = values.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return values.map((row) => row.map(() => 0.5));
  }
  return values.map((row) => row.map((value) => (value - min) / (max - min)));
}
