import { useState } from "react";

export function useStored<T>(key: string, fallback: () => T, parse: (raw: string) => T) {
  const [initial] = useState(() => {
    if (typeof window === "undefined") return { value: fallback(), error: "", writable: true };
    try {
      const raw = localStorage.getItem(key);
      return { value: raw === null ? fallback() : parse(raw), error: "", writable: true };
    } catch (error) {
      return { value: fallback(), error: String(error), writable: false };
    }
  });
  const [value, setValue] = useState(initial.value);
  const [errors, setErrors] = useState<string[]>(initial.error ? [initial.error] : []);
  const update = (next: T) => {
    if (!initial.writable) {
      setErrors(previous => [...previous, "Storage recovery required. Original data was not overwritten; edit was not applied."]);
      return false;
    }
    try {
      localStorage.setItem(key, JSON.stringify(next));
      setValue(next);
      return true;
    } catch (error) {
      setErrors(previous => [...previous, `Edit was not saved or applied: ${String(error)}`]);
      return false;
    }
  };
  return { value, update, errors };
}
