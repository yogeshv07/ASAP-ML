/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#e6eef8",
        midnight: "#0b0f19",
        slatecore: "#0f172a",
        tealglow: "#14b8a6",
        blueglow: "#4f46e5",
        glassline: "rgba(148, 163, 184, 0.18)"
      },
      boxShadow: {
        glass: "0 24px 80px rgba(2, 6, 23, 0.45)",
        glow: "0 0 0 1px rgba(20,184,166,0.12), 0 18px 40px rgba(20,184,166,0.18)",
        violet: "0 18px 40px rgba(79,70,229,0.2)"
      },
      backgroundImage: {
        "hero-grid":
          "radial-gradient(circle at 20% 20%, rgba(20,184,166,0.18), transparent 22%), radial-gradient(circle at 80% 0%, rgba(79,70,229,0.14), transparent 28%), linear-gradient(180deg, rgba(15,23,42,0.96), rgba(11,15,25,1))"
      },
      animation: {
        float: "float 6s ease-in-out infinite"
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" }
        }
      }
    }
  },
  plugins: []
};
