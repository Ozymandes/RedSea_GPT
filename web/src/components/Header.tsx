import { WaveIcon, ResetIcon } from "./Icons";
import { ToneToggle, type Tone } from "./ToneToggle";

interface Props {
  engine: string;
  onReset: () => void;
  hasMessages: boolean;
  tone: Tone;
  onToneChange: (t: Tone) => void;
}

export function Header({ engine, onReset, hasMessages, tone, onToneChange }: Props) {
  return (
    <header className="sticky top-0 z-20 border-b hairline bg-bone/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-3xl items-center justify-between px-5">
        <div className="flex items-center gap-2.5">
          <span className="text-ink">
            <WaveIcon width={20} height={20} />
          </span>
          <div className="flex items-baseline gap-2">
            <span className="font-editorial text-[19px] font-medium text-ink-950">
              RedSea
            </span>
            <span className="mono-label">GPT</span>
          </div>
          <span className="mono-label hidden text-[9px] text-muted/60 sm:inline" title={`Engine: ${engine === "agent" ? "CRAG agent" : "baseline RAG"}`}>
            {engine === "agent" ? "CRAG" : "RAG"}
          </span>
        </div>
        <div className="flex items-center gap-2.5">
          <ToneToggle tone={tone} onChange={onToneChange} />
          {hasMessages && (
            <button
              onClick={onReset}
              className="pressable inline-flex items-center gap-1.5 rounded-md border hairline bg-white px-2.5 py-1.5 text-[12px] font-medium text-ink/70 hover:text-ink hover:bg-bone-50"
              aria-label="Reset conversation"
            >
              <ResetIcon width={13} height={13} />
              <span className="hidden sm:inline">New chat</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
