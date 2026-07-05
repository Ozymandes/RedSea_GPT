/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Red Sea palette — synthesized from minimalist-ui + brand adaptation.
        // Deep teal is the "ink"; coral is the scarce semantic accent.
        bone: {
          DEFAULT: "#F7F6F3", // canvas / paper
          50: "#FBFBFA",
          100: "#F2F1ED",
          200: "#EAE9E4",
        },
        ink: {
          DEFAULT: "#0E3B43", // deep teal — primary brand
          950: "#0A2E34",
          700: "#155059",
          500: "#2A6B75",
        },
        coral: {
          DEFAULT: "#E07856", // the single scarce accent
          soft: "#F4D7CB",
          deep: "#B85A3B",
        },
        line: "#EAEAEA", // 1px borders everywhere
        muted: "#6B7480",
      },
      fontFamily: {
        // Inter/Roboto/Open Sans are explicitly banned by minimalist-ui.
        serif: ['"Newsreader"', '"Instrument Serif"', "Georgia", "serif"],
        sans: [
          '"Geist"',
          '"SF Pro Display"',
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
        mono: ['"JetBrains Mono"', '"Geist Mono"', '"SF Mono"', "monospace"],
      },
      letterSpacing: {
        tightest: "-0.04em",
      },
      maxWidth: {
        prose: "44rem",
      },
      transitionTimingFunction: {
        // Kowalski: custom ease-out, never the weak built-ins.
        "out-expo": "cubic-bezier(0.23, 1, 0.32, 1)",
        "in-out-expo": "cubic-bezier(0.77, 0, 0.175, 1)",
      },
    },
  },
  plugins: [],
};
