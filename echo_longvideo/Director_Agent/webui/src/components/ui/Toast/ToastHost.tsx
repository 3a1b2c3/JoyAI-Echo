import classNames from "classnames";
import { useCallback, useEffect, useState, type TransitionEvent } from "react";
import {
  registerToastListener,
  type ToastState,
  type ToastType,
} from "./toast";
import styles from "./Toast.module.scss";

function ToastIcon({ type }: { type: ToastType }) {
  if (type === "success") {
    return (
      <div className={styles.toastIconSuccess} aria-hidden="true">
        <svg
          viewBox="64 64 896 896"
          focusable="false"
          width="1em"
          height="1em"
          fill="currentColor"
        >
          <path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm193.5 301.7l-210.6 292a31.8 31.8 0 01-51.7 0L318.5 484.9c-3.8-5.3 0-12.7 6.5-12.7h46.9c10.2 0 19.9 4.9 25.9 13.3l71.2 98.8 157.2-218c6-8.3 15.6-13.3 25.9-13.3H699c6.5 0 10.3 7.4 6.5 12.7z" />
        </svg>
      </div>
    );
  }

  return <div className={styles.toastIcon}>!</div>;
}

/** 全局 Toast 挂载点，配合 toast.error / toast.success 使用 */
export function ToastHost() {
  const [toastState, setToastState] = useState<ToastState | null>(null);

  useEffect(() => registerToastListener(setToastState), []);

  const handleTransitionEnd = useCallback((event: TransitionEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return;
    if (event.propertyName !== "opacity") return;

    setToastState((prev) => {
      if (prev && !prev.visible) return null;
      return prev;
    });
  }, []);

  return (
    <div
      className={classNames(
        styles.toast,
        toastState?.visible && styles.visible,
      )}
      role="status"
      aria-live="polite"
      onTransitionEnd={handleTransitionEnd}
    >
      {toastState && (
        <>
          <ToastIcon type={toastState.type} />
          <span className={styles.toastText}>{toastState.message}</span>
        </>
      )}
    </div>
  );
}
