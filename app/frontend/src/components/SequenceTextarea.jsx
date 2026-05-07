function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderHighlightedSequence(value) {
  const colorMap = {
    A: "text-emerald-300",
    T: "text-sky-300",
    G: "text-violet-300",
    C: "text-amber-300"
  };

  return escapeHtml(value || "Paste DNA sequence here")
    .split("")
    .map((character) => {
      if (character === "\n") {
        return "<br />";
      }
      const upper = character.toUpperCase();
      const cssClass = colorMap[upper] || "text-slate-500";
      const safeCharacter = character === " " ? "&nbsp;" : escapeHtml(character);
      return `<span class="${cssClass}">${safeCharacter}</span>`;
    })
    .join("");
}

export function SequenceTextarea({ value, onChange }) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-slate-700/70 bg-slate-950/70">
      <pre
        aria-hidden="true"
        className="pointer-events-none min-h-[280px] whitespace-pre-wrap break-words px-4 py-4 font-mono text-sm leading-7 tracking-[0.08em]"
        dangerouslySetInnerHTML={{ __html: renderHighlightedSequence(value) }}
      />
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        spellCheck={false}
        className="absolute inset-0 h-full w-full resize-none bg-transparent px-4 py-4 font-mono text-sm leading-7 tracking-[0.08em] text-transparent caret-emerald-300 outline-none"
      />
    </div>
  );
}
