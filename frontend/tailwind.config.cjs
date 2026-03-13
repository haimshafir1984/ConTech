/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: { DEFAULT: "#0D1F3C", 2: "#1B3A6B" },
        blue: { DEFAULT: "#2563EB", hover: "#1D4ED8", 50: "#EFF6FF", 100: "#DBEAFE" },
        green: { DEFAULT: "#059669", 50: "#ECFDF5" },
        amber: { DEFAULT: "#D97706", 50: "#FFFBEB" },
        red: { DEFAULT: "#DC2626", 50: "#FEF2F2" },
        s: {
          50: "#F8FAFC", 100: "#F1F5F9", 200: "#E2E8F0",
          300: "#CBD5E1", 400: "#94A3B8", 500: "#64748B",
          700: "#334155", 900: "#0F172A",
        },
      },
      boxShadow: {
        sh1: "0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04)",
        sh2: "0 4px 16px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.03)",
      },
      borderRadius: {
        r: "12px",
        "r-sm": "8px",
      },
    }
  },
  plugins: []
};

