/** Renders `**bold**` segments of a plain-text string as <strong>. No HTML is ever injected. */
export default function FormattedText({ text }: { text: string }) {
  const BOLD_RE = /\*\*(.+?)\*\*/g;
  const parts: Array<{ text: string; bold: boolean }> = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = BOLD_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ text: text.slice(lastIndex, match.index), bold: false });
    }
    parts.push({ text: match[1], bold: true });
    lastIndex = BOLD_RE.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push({ text: text.slice(lastIndex), bold: false });
  }

  return (
    <>
      {parts.map((part, i) =>
        part.bold ? (
          <strong key={i} className="font-semibold text-white">
            {part.text}
          </strong>
        ) : (
          <span key={i}>{part.text}</span>
        )
      )}
    </>
  );
}
