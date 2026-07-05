import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

// Render the assistant answer safely with react-markdown (no dangerouslySetInnerHTML,
// ever) AND preserve the clickable [n] citation chips inside the prose.
//
// Approach: preprocess the raw answer to turn each [n] citation into a markdown
// link of the form [#cite-<n>]. react-markdown parses those as <a> nodes; we
// override the `a` renderer to detect the #cite- scheme and render our chip
// button instead of a real link. Safe (no raw HTML reaches the DOM) and the
// chip stays interactive (scroll-to + highlight the source card).

interface Props {
  text: string;
  onChip: (id: number) => void;
}

// Match [n] or [n][m] runs. Convert each [n] -> [\1](#cite-\1).
const CITE_RE = /\[(\d{1,2})\]/g;

function preprocess(text: string): string {
  return text.replace(CITE_RE, (_m, n: string) => `[${n}](#cite-${n})`);
}

export function AnswerBody({ text, onChip }: Props) {
  const processed = useMemo(() => preprocess(text), [text]);

  const components: Components = useMemo(
    () => ({
      // Citation chips: render as the interactive chip, not a link.
      a({ href, children }) {
        const m = /^#cite-(\d{1,2})$/.exec(href || "");
        if (m) {
          const id = parseInt(m[1], 10);
          return (
            <button
              className="cite-chip"
              onClick={(e) => {
                e.preventDefault();
                onChip(id);
              }}
              aria-label={`See source ${id}`}
            >
              {id}
            </button>
          );
        }
        // Real links (rare) — render plainly, open in new tab.
        return (
          <a href={href} target="_blank" rel="noopener noreferrer">
            {children}
          </a>
        );
      },
      strong({ children }) {
        return <strong className="font-semibold text-ink-950">{children}</strong>;
      },
      em({ children }) {
        return <em className="italic">{children}</em>;
      },
      // Drop the outer <p> wrapper react-markdown adds so the prose spacing
      // in index.css (.answer-prose p) controls margins cleanly.
      p({ children }) {
        return <p>{children}</p>;
      },
      h3({ children }) {
        return <h3>{children}</h3>;
      },
      h4({ children }) {
        return <h4>{children}</h4>;
      },
      ul({ children }) {
        return <ul className="my-2 list-disc pl-5 space-y-1">{children}</ul>;
      },
      ol({ children }) {
        return <ol className="my-2 list-decimal pl-5 space-y-1">{children}</ol>;
      },
      li({ children }) {
        return <li className="leading-relaxed">{children}</li>;
      },
      code({ children }) {
        return (
          <code className="rounded bg-bone-100 px-1 py-0.5 font-mono text-[13px] text-ink-950">
            {children}
          </code>
        );
      },
      // Disable raw HTML entirely (defense in depth; react-markdown already does).
      // react-markdown v9 doesn't render raw HTML by default, so this is a no-op
      // belt-and-braces.
    }),
    [onChip]
  );

  return (
    <div className="answer-prose">
      <ReactMarkdown components={components}>{processed}</ReactMarkdown>
    </div>
  );
}
