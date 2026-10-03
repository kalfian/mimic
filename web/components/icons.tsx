/**
 * One icon family: 16 px grid, 1.5 px stroke, round caps, currentColor. Decorative by default
 * (aria-hidden); the surrounding control carries the accessible name.
 */

import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 16, children, ...rest }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconCheck = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3.5 8.5l3 3 6-7" />
  </Svg>
);

export const IconCross = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 4l8 8M12 4l-8 8" />
  </Svg>
);

export const IconAlert = (p: IconProps) => (
  <Svg {...p}>
    <path d="M8 2.5l6 10.5H2L8 2.5z" />
    <path d="M8 6.5v3" />
    <path d="M8 11.5v.01" strokeWidth={2} />
  </Svg>
);

export const IconInfo = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M8 7.25V11" />
    <path d="M8 5v.01" strokeWidth={2} />
  </Svg>
);

export const IconUpload = (p: IconProps) => (
  <Svg {...p}>
    <path d="M8 10V2.5M5 5.5l3-3 3 3" />
    <path d="M2.5 9.5v3a1 1 0 001 1h9a1 1 0 001-1v-3" />
  </Svg>
);

export const IconFilm = (p: IconProps) => (
  <Svg {...p}>
    <rect x="2" y="3" width="12" height="10" rx="1.5" />
    <path d="M5 3v10M11 3v10M2 6h3M2 10h3M11 6h3M11 10h3" />
  </Svg>
);

export const IconCopy = (p: IconProps) => (
  <Svg {...p}>
    <rect x="5.5" y="5.5" width="8" height="8" rx="1.5" />
    <path d="M10.5 5.5V3.5a1 1 0 00-1-1h-6a1 1 0 00-1 1v6a1 1 0 001 1h2" />
  </Svg>
);

export const IconPlay = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 3.5v9l7.5-4.5L5 3.5z" fill="currentColor" />
  </Svg>
);

export const IconPause = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5.5 3.5v9M10.5 3.5v9" strokeWidth={2} />
  </Svg>
);

export const IconStepBack = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 3.5v9" />
    <path d="M12 3.5v9L6.5 8 12 3.5z" fill="currentColor" />
  </Svg>
);

export const IconStepForward = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 3.5v9" />
    <path d="M4 3.5v9L9.5 8 4 3.5z" fill="currentColor" />
  </Svg>
);

export const IconArrowLeft = (p: IconProps) => (
  <Svg {...p}>
    <path d="M13 8H3M7 4L3 8l4 4" />
  </Svg>
);

export const IconRefresh = (p: IconProps) => (
  <Svg {...p}>
    <path d="M13 8a5 5 0 11-1.46-3.54" />
    <path d="M13 2.5v3h-3" />
  </Svg>
);

/** Indeterminate activity. Spins only when motion is allowed (globals.css reduced-motion rule). */
export const IconSpinner = ({ className = "", ...p }: IconProps) => (
  <Svg {...p} className={`spin ${className}`}>
    <circle cx="8" cy="8" r="5.5" opacity={0.25} />
    <path d="M13.5 8A5.5 5.5 0 008 2.5" />
  </Svg>
);

/** Wordmark glyph: a box and its displaced ghost joined by a motion path (state A → state B). */
export const IconMark = (p: IconProps) => (
  <Svg {...p} viewBox="0 0 20 20">
    <rect x="2" y="9" width="7" height="7" rx="1.25" strokeDasharray="1.5 1.75" />
    <rect x="11" y="4" width="7" height="7" rx="1.25" fill="currentColor" stroke="none" />
    <path d="M9 12.5c2 0 2.5-3 2.5-3" />
  </Svg>
);
