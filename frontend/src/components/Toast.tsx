import React from "react";

type ToastType = "success" | "error" | "info";

interface ToastItem {
  id: number;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  toast: (message: string, type?: ToastType) => void;
}

const ToastContext = React.createContext<ToastContextValue>({ toast: () => {} });

let _nextId = 0;

const COLORS: Record<ToastType, { bg: string; border: string; icon: string }> = {
  success: { bg: "#F0FDF4", border: "#16A34A", icon: "✅" },
  error:   { bg: "#FEF2F2", border: "#DC2626", icon: "❌" },
  info:    { bg: "#EFF6FF", border: "#2563EB", icon: "ℹ️" },
};

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = React.useState<ToastItem[]>([]);

  const toast = React.useCallback((message: string, type: ToastType = "success") => {
    const id = ++_nextId;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  }, []);

  const remove = (id: number) =>
    setToasts((prev) => prev.filter((t) => t.id !== id));

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div
        style={{
          position: "fixed",
          top: 68,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 9999,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          alignItems: "center",
          pointerEvents: "none",
        }}
      >
        {toasts.map((t) => {
          const c = COLORS[t.type];
          return (
            <div
              key={t.id}
              onClick={() => remove(t.id)}
              style={{
                background: c.bg,
                border: `1.5px solid ${c.border}`,
                borderRadius: 10,
                padding: "10px 20px",
                display: "flex",
                alignItems: "center",
                gap: 10,
                boxShadow: "0 4px 16px rgba(0,0,0,0.13)",
                fontSize: 14,
                fontWeight: 500,
                color: "#1f2937",
                minWidth: 220,
                maxWidth: 420,
                pointerEvents: "auto",
                direction: "rtl",
                cursor: "pointer",
                animation: "toastIn 0.22s ease",
              }}
            >
              <span style={{ fontSize: 18, flexShrink: 0 }}>{c.icon}</span>
              <span style={{ flex: 1 }}>{t.message}</span>
              <span style={{ color: "#9CA3AF", fontSize: 16, marginRight: 4 }}>×</span>
            </div>
          );
        })}
      </div>
      <style>{`
        @keyframes toastIn {
          from { opacity: 0; transform: translateY(-10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </ToastContext.Provider>
  );
};

export const useToast = (): ((message: string, type?: ToastType) => void) =>
  React.useContext(ToastContext).toast;
