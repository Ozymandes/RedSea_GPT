import { WaveIcon, ResetIcon, MenuIcon } from "./Icons";
import { ToneToggle, type Tone } from "./ToneToggle";

interface Props {
  engine: string;
  onReset: () => void;
  hasMessages: boolean;
  tone: Tone;
  onToneChange: (t: Tone) => void;
  onOpenSidebar: () => void;
  sidebarOpen: boolean;
}

export function Header({
  engine,
  onReset,
  hasMessages,
  tone,
  onToneChange,
  onOpenSidebar,
  sidebarOpen,
}: Props) {
  return (
    <header className="sticky top-0 z-20 border-b hairline bg-bone/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-3xl items-center gap-2 px-5">
        {/* Menu button — visible whenever the sidebar is CLOSED (all screen sizes),
            so a desktop user who collapsed the sidebar can reopen it. Hidden on
            desktop when open (the sidebar shows its own collapse chevron then). */}
        <button
          onClick={onOpenSidebar}
          className={`pressable -ml-1 rounded-md p-1.5 text-ink/70 hover:bg-bone-100 hover:text-ink-950 ${
            sidebarOpen ? "md:hidden" : ""
          }`}
          aria-label="Open chat history"
        >
          <MenuIcon width={18} height={18} />
        </button>

        <div className="flex min-w-0 flex-1 items-center gap-2.5">
          <span className="text-ink md:hidden">
            <WaveIcon width={18} height={18} />
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
              aria-label="Start new chat"
            >
              <ResetIcon width={13} height={13} />
              <span className="hidden sm:inline">New</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
