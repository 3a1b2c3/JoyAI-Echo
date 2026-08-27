export type ToastType = "error" | "success";

export interface ToastOptions {
  message: string;
  type?: ToastType;
  duration?: number;
}

export interface ToastState {
  message: string;
  type: ToastType;
  visible: boolean;
}

type ToastListener = (state: ToastState | null) => void;

const DEFAULT_DURATION = 3000;

let listener: ToastListener | null = null;
let timer: number | null = null;
let currentToast: ToastState | null = null;

function clearTimer() {
  if (timer !== null) {
    window.clearTimeout(timer);
    timer = null;
  }
}

function emit(state: ToastState | null) {
  listener?.(state);
}

function normalizeOptions(
  options: ToastOptions | string,
  type?: ToastType,
  duration?: number,
): Required<Pick<ToastOptions, "message" | "type" | "duration">> {
  if (typeof options === "string") {
    return {
      message: options,
      type: type ?? "error",
      duration: duration ?? DEFAULT_DURATION,
    };
  }
  return {
    message: options.message,
    type: options.type ?? "error",
    duration: options.duration ?? DEFAULT_DURATION,
  };
}

function showToast(options: ToastOptions | string) {
  const { message, type, duration } = normalizeOptions(options);

  clearTimer();
  currentToast = { message, type, visible: true };
  emit(currentToast);

  timer = window.setTimeout(() => {
    if (currentToast) {
      currentToast = { ...currentToast, visible: false };
      emit(currentToast);
    }
    timer = null;
  }, duration);
}

export function registerToastListener(fn: ToastListener): () => void {
  listener = fn;
  return () => {
    if (listener === fn) {
      listener = null;
    }
  };
}

export const toast = {
  show(options: ToastOptions | string) {
    showToast(options);
  },
  error(message: string, duration?: number) {
    showToast({ message, type: "error", duration });
  },
  success(message: string, duration?: number) {
    showToast({ message, type: "success", duration });
  },
};
