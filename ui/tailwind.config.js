/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        ink: {
          50: "#f7f8fa",
          100: "#eef0f4",
          200: "#dde1e9",
          300: "#bcc3d1",
          400: "#8d97aa",
          500: "#5b6478",
          600: "#3f4757",
          700: "#2c3340",
          800: "#1d222d",
          900: "#101319",
        },
        accent: {
          DEFAULT: "#4F46E5",
          soft: "#EEF2FF",
          ring: "#C7D2FE",
        },
      },
      boxShadow: {
        card: "0 1px 2px rgba(16,19,25,0.04), 0 1px 1px rgba(16,19,25,0.02)",
        cardHover: "0 4px 12px rgba(16,19,25,0.06), 0 1px 2px rgba(16,19,25,0.04)",
      },
    },
  },
  plugins: [],
};
