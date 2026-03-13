import React from "react";

interface ConfirmOptions {
  title?: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  danger?: boolean;
}

interface ConfirmContextValue {
  confirm: (options: ConfirmOptions) => Promise<boolean>;
}

const ConfirmContext = React.createContext<ConfirmContextValue>({
  confirm: async () => false,
});

export const ConfirmProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = React.useState<{
    options: ConfirmOptions;
    resolve: (v: boolean) => void;
  } | null>(null);

  const confirm = React.useCallback((options: ConfirmOptions): Promise<boolean> => {
    return new Promise((resolve) => {
      setState({ options, resolve });
    });
  }, []);

  const handleClose = (result: boolean) => {
    state?.resolve(result);
    setState(null);
  };

  return (
    <ConfirmContext.Provider value={{ confirm }}>
      {children}
      {state && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.45)",
            zIndex: 10000,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          onClick={() => handleClose(false)}
        >
          <div
            style={{
              background: "#fff",
              borderRadius: 14,
              padding: "28px 32px",
              maxWidth: 400,
              width: "90%",
              boxShadow: "0 20px 60px rgba(0,0,0,0.22)",
              direction: "rtl",
              animation: "confirmIn 0.18s ease",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {state.options.title && (
              <div
                style={{
                  fontWeight: 700,
                  fontSize: 17,
                  marginBottom: 10,
                  color: state.options.danger ? "#DC2626" : "#1B3A6B",
                }}
              >
                {state.options.title}
              </div>
            )}
            <div
              style={{
                fontSize: 14,
                color: "#374151",
                lineHeight: 1.65,
                marginBottom: 24,
              }}
            >
              {state.options.message}
            </div>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button
                type="button"
                onClick={() => handleClose(false)}
                style={{
                  padding: "8px 22px",
                  border: "1.5px solid #D1D5DB",
                  borderRadius: 8,
                  background: "#fff",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 500,
                  color: "#374151",
                }}
              >
                {state.options.cancelText ?? "ביטול"}
              </button>
              <button
                type="button"
                onClick={() => handleClose(true)}
                style={{
                  padding: "8px 22px",
                  border: "none",
                  borderRadius: 8,
                  background: state.options.danger ? "#DC2626" : "#1B3A6B",
                  color: "#fff",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 600,
                }}
              >
                {state.options.confirmText ?? "אישור"}
              </button>
            </div>
          </div>
        </div>
      )}
      <style>{`
        @keyframes confirmIn {
          from { opacity: 0; transform: scale(0.95); }
          to   { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </ConfirmContext.Provider>
  );
};

export const useConfirm = (): ((options: ConfirmOptions) => Promise<boolean>) =>
  React.useContext(ConfirmContext).confirm;
