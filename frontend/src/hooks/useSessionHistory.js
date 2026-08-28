import { useCallback, useState } from "react";

const KEY = "skycast.session.predictions";

function read() {
  try {
    return JSON.parse(sessionStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

export function useSessionHistory() {
  const [items, setItems] = useState(() => read());

  const add = useCallback((entry) => {
    setItems((prev) => {
      const next = [{ ...entry, savedAt: new Date().toISOString() }, ...prev].slice(0, 20);
      sessionStorage.setItem(KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    sessionStorage.removeItem(KEY);
    setItems([]);
  }, []);

  return { items, add, clear };
}
