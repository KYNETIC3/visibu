import type { ReactElement } from 'react'

const paths: Record<string, ReactElement> = {
  hand: (
    <path d="M6 9V4.5a1.2 1.2 0 0 1 2.4 0V8M8.4 8V3.2a1.2 1.2 0 0 1 2.4 0V8M10.8 8V4.2a1.2 1.2 0 0 1 2.4 0V10c0 3-2 5-5 5-2 0-3.2-1-4.4-2.8L2.6 10a1.1 1.1 0 0 1 1.8-1.3L6 10.5" />
  ),
  mic: (
    <>
      <rect x="6.5" y="2" width="5" height="9" rx="2.5" />
      <path d="M4 8.5a5 5 0 0 0 10 0M9 13.5V16" />
    </>
  ),
  keyboard: (
    <>
      <rect x="2" y="4.5" width="14" height="9" rx="2" />
      <path d="M5 7.5h.01M8 7.5h.01M11 7.5h.01M13 7.5h.01M5.5 10.5h7" />
    </>
  ),
  check: <path d="m4 9.5 3 3L14 5.5" />,
  alert: <path d="M9 4v6M9 13v.5" />,
  pause: (
    <>
      <rect x="4" y="3.5" width="3.5" height="11" rx="1" />
      <rect x="10.5" y="3.5" width="3.5" height="11" rx="1" />
    </>
  ),
  play: <path d="M5 3.5v11l9-5.5-9-5.5Z" />,
  mirror: <path d="M9 2v14M3 5l3 4-3 4M15 5l-3 4 3 4" />,
  screen: (
    <>
      <rect x="2" y="3" width="14" height="10" rx="2" />
      <path d="M6 16h6" />
    </>
  ),
  copy: (
    <>
      <rect x="6" y="6" width="9" height="9" rx="2" />
      <path d="M12 6V4.5A1.5 1.5 0 0 0 10.5 3h-6A1.5 1.5 0 0 0 3 4.5v6A1.5 1.5 0 0 0 4.5 12H6" />
    </>
  ),
  close: <path d="M4 4l10 10M14 4 4 14" />,
  search: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <path d="m12 12 4 4" />
    </>
  ),
  share: <path d="M9 11V2.5M5.5 6 9 2.5 12.5 6M3 11v3.5h12V11" />,
  trash: <path d="M3.5 5h11M7 5V3.5h4V5M5 5l.8 9.5h6.4L13 5" />,
  camera: (
    <>
      <rect x="2" y="4.5" width="10.5" height="9" rx="2" />
      <path d="m12.5 8 3.5-2v6l-3.5-2" />
    </>
  ),
  refresh: (
    <>
      <circle cx="9" cy="9" r="6.5" />
      <path d="M9 5.5V9l2.5 1.5" />
    </>
  ),
  chevron: <path d="m5 7 4 4 4-4" />,
  send: <path d="M3 9h11M10 5l4 4-4 4" />,
  arrowUp: <path d="M9 15V3M4 8l5-5 5 5" />,
  gear: (
    <>
      <circle cx="9" cy="9" r="2.4" />
      <path d="M9 1.8v2M9 14.2v2M1.8 9h2M14.2 9h2M3.9 3.9l1.4 1.4M12.7 12.7l1.4 1.4M3.9 14.1l1.4-1.4M12.7 5.3l1.4-1.4" />
      <circle cx="9" cy="9" r="5.2" />
    </>
  ),
  clock: (
    <>
      <circle cx="9" cy="9" r="7" />
      <path d="M9 5v4l2.5 1.6" />
    </>
  ),
  bookmark: <path d="M13.5 15.5 9 12.2l-4.5 3.3V3.8A1.3 1.3 0 0 1 5.8 2.5h6.4a1.3 1.3 0 0 1 1.3 1.3Z" />,
  bubble: <path d="M15.5 11a1.5 1.5 0 0 1-1.5 1.5H6l-3.5 3V4A1.5 1.5 0 0 1 4 2.5h10A1.5 1.5 0 0 1 15.5 4Z" />,
  pencil: <path d="M9.5 15h6M12.3 2.7a1.6 1.6 0 0 1 2.3 2.3L5.5 14.1l-3 .8.8-3Z" />,
  chevronRight: <path d="m7 4 5 5-5 5" />,
  chevronLeft: <path d="m11 4-5 5 5 5" />,
  sun: (
    <>
      <circle cx="9" cy="9" r="3" />
      <path d="M9 1.5v1.5M9 15v1.5M1.5 9H3M15 9h1.5M3.7 3.7l1 1M13.3 13.3l1 1M3.7 14.3l1-1M13.3 4.7l1-1" />
    </>
  ),
  lock: (
    <>
      <rect x="3.5" y="8" width="11" height="8" rx="2" />
      <path d="M6 8V5.5a3 3 0 0 1 6 0V8" />
    </>
  ),
  person: (
    <>
      <circle cx="9" cy="6" r="3" />
      <path d="M3 16v-.5A4.5 4.5 0 0 1 7.5 11h3a4.5 4.5 0 0 1 4.5 4.5v.5" />
    </>
  ),
}

export type IconName = keyof typeof paths

interface Props {
  name: IconName
  size?: number
  strokeWidth?: number
  className?: string
}

/** Decorative stroke icon. Always pair it with visible text or an aria-label on the control. */
export function Icon({ name, size = 18, strokeWidth = 1.6, className }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 18 18"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className={className}
    >
      {paths[name]}
    </svg>
  )
}

export function Logo() {
  return (
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden="true" focusable="false">
      <rect width="36" height="36" rx="11" fill="#17181A" />
      <path d="M8 18c2.8-5 6.2-7.5 10-7.5S25.2 13 28 18c-2.8 5-6.2 7.5-10 7.5S10.8 23 8 18Z" stroke="#FFFFFF" strokeWidth="2" strokeLinejoin="round" />
      <circle cx="18" cy="18" r="3.2" fill="#8EA2F0" />
    </svg>
  )
}
