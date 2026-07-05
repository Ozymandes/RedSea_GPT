import { useEffect, useRef, useState } from "react";
import { SendIcon } from "./Icons";

interface Props {
  onSend: (text: string) => void;
  disabled: boolean;
}

export function Composer({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Autosize the textarea (only transform/height; bounded).
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "0px";
    ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
  }, [value]);

  const submit = () => {
    const t = value.trim();
    if (!t || disabled) return;
    onSend(t);
    setValue("");
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t hairline bg-bone/80 backdrop-blur-md">
      <div className="mx-auto max-w-3xl px-5 py-3.5">
        <div className="pressable flex items-end gap-2 rounded-xl border hairline bg-white p-2 focus-within:border-ink/30">
          <textarea
            ref={taRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={onKey}
            rows={1}
            placeholder="Ask about the Red Sea — geology, oceanography, reefs, biodiversity…"
            className="max-h-[180px] flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed text-ink-950 outline-none placeholder:text-muted/60"
          />
          <button
            onClick={submit}
            disabled={disabled || !value.trim()}
            className="pressable inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink text-bone hover:bg-ink-700 disabled:cursor-not-allowed disabled:opacity-30"
            aria-label="Send"
          >
            <SendIcon width={16} height={16} />
          </button>
        </div>
        <p className="mt-2 px-1 text-[11px] text-muted/70">
          Cites its sources. Refuses what it can&rsquo;t ground.
        </p>
      </div>
    </div>
  );
}
