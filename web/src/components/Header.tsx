import { WaveIcon, ResetIcon } from "./Icons";

interface Props {
  engine: string;
  onReset: () => void;
  hasMessages: boolean;
}

export function Header({ engine, onReset, hasMessages }: Props) {
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
        </div>
        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-1.5 sm:inline-flex">
            <span className="h-1.5 w-1.5 rounded-full bg-coral" />
            <span className="mono-label">
              {engine === "agent" ? "CRAG agent" : "baseline RAG"}
            </span>
          </span>
          {hasMessages && (
            <button
              onClick={onReset}
              className="pressable inline-flex items-center gap-1.5 rounded-md border hairline bg-white px-2.5 py-1.5 text-[12px] font-medium text-ink/70 hover:text-ink hover:bg-bone-50"
              aria-label="Reset conversation"
            >
              <ResetIcon width={13} height={13} />
              New chat
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
