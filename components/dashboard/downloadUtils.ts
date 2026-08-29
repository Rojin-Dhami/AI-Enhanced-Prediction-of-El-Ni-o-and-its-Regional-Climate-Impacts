import { toPng } from "html-to-image";

function triggerDownload(dataUrl: string, filename: string) {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  link.click();
}

export async function downloadPng(
  ref: HTMLElement,
  filename: string,
  bgColor = "#0f172a"
) {
  try {
    const dataUrl = await toPng(ref, { backgroundColor: bgColor, pixelRatio: 2 });
    triggerDownload(dataUrl, filename);
  } catch (err) {
    console.error("Failed to export PNG:", err);
  }
}

function objectToCsvRow(obj: Record<string, unknown>): string {
  return Object.values(obj)
    .map((v) => {
      const s = String(v ?? "");
      return s.includes(",") || s.includes('"') || s.includes("\n")
        ? `"${s.replace(/"/g, '""')}"`
        : s;
    })
    .join(",");
}

export function downloadCsv(
  rows: Record<string, unknown>[],
  filename: string
) {
  if (!rows.length) return;
  const header = Object.keys(rows[0]).join(",");
  const body = rows.map(objectToCsvRow).join("\n");
  const csv = `${header}\n${body}`;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  triggerDownload(url, filename);
  URL.revokeObjectURL(url);
}

export function downloadJson(data: unknown, filename: string) {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  triggerDownload(url, filename);
  URL.revokeObjectURL(url);
}
