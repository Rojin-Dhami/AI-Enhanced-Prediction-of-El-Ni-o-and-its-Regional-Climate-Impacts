"use client";

import { useState, useRef, useEffect, ReactNode } from "react";
import { createPortal } from "react-dom";

interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  position?: "top" | "bottom" | "left" | "right";
  delay?: number;
}

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

export default function Tooltip({
  content,
  children,
  position = "top",
  delay = 150,
}: TooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [rect, setRect] = useState<Rect | null>(null);
  const triggerRef = useRef<HTMLElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const open = () => {
    timeoutRef.current = setTimeout(() => {
      if (triggerRef.current) {
        const r = triggerRef.current.getBoundingClientRect();
        setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
      }
      setIsOpen(true);
    }, delay);
  };

  const close = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsOpen(false);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen]);

  const tooltipStyle: React.CSSProperties = {
    position: "fixed",
    zIndex: 9999,
    maxWidth: 320,
    padding: "10px 12px",
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 8,
    color: "#e2e8f0",
    fontSize: 12,
    lineHeight: 1.5,
    boxShadow: "0 10px 30px rgba(0,0,0,0.4)",
    pointerEvents: "none",
  };

  const arrowStyle: React.CSSProperties = {
    position: "absolute",
    width: 0,
    height: 0,
    borderWidth: 6,
    borderStyle: "solid",
  };

  const positions = {
    top: {
      tooltip: { ...tooltipStyle, bottom: "calc(100% + 8px)", left: "50%", transform: "translateX(-50%)" },
      arrow: { ...arrowStyle, bottom: -12, left: "50%", marginLeft: -6, borderColor: "#0f172a transparent transparent transparent", borderTopWidth: 6, borderBottomWidth: 0 },
    },
    bottom: {
      tooltip: { ...tooltipStyle, top: "calc(100% + 8px)", left: "50%", transform: "translateX(-50%)" },
      arrow: { ...arrowStyle, top: -12, left: "50%", marginLeft: -6, borderColor: "transparent transparent #0f172a transparent", borderBottomWidth: 6, borderTopWidth: 0 },
    },
    left: {
      tooltip: { ...tooltipStyle, right: "calc(100% + 8px)", top: "50%", transform: "translateY(-50%)" },
      arrow: { ...arrowStyle, left: -12, top: "50%", marginTop: -6, borderColor: "transparent transparent transparent #0f172a", borderLeftWidth: 6, borderRightWidth: 0 },
    },
    right: {
      tooltip: { ...tooltipStyle, left: "calc(100% + 8px)", top: "50%", transform: "translateY(-50%)" },
      arrow: { ...arrowStyle, right: -12, top: "50%", marginTop: -6, borderColor: "transparent #0f172a transparent transparent", borderRightWidth: 6, borderLeftWidth: 0 },
    },
  };

  const pos = positions[position];

  if (!isOpen || !rect) return <>{children}</>;

  const styles = { ...pos.tooltip } as React.CSSProperties;
  const tooltipHeight = 80;

  if (position === "top") {
    styles.bottom = "calc(100% + 8px)";
    styles.left = "50%";
    styles.transform = "translateX(-50%)";
    styles.top = undefined;
  } else if (position === "bottom") {
    styles.top = "calc(100% + 8px)";
    styles.left = "50%";
    styles.transform = "translateX(-50%)";
    styles.bottom = undefined;
  } else if (position === "left") {
    styles.right = "calc(100% + 8px)";
    styles.top = "50%";
    styles.transform = "translateY(-50%)";
    styles.left = undefined;
  } else if (position === "right") {
    styles.left = "calc(100% + 8px)";
    styles.top = "50%";
    styles.transform = "translateY(-50%)";
    styles.right = undefined;
  }

  styles.top = `${rect.top - tooltipHeight - 8}px`;
  styles.left = `${rect.left + rect.width / 2}px`;

  const portal = createPortal(
    <div style={styles} role="tooltip" id="tooltip-content">
      {content}
      <div style={pos.arrow} />
    </div>,
    document.body
  );

  return (
    <span
      ref={triggerRef}
      onMouseEnter={open}
      onMouseLeave={close}
      onFocus={open}
      onBlur={close}
    >
      {children}
      {isOpen && rect && portal}
    </span>
  );
}