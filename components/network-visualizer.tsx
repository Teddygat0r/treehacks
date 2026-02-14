"use client"

import { motion } from "framer-motion"

export type NetworkPhase = "idle" | "drafting" | "verifying" | "correcting" | "complete"

interface NetworkVisualizerProps {
  phase: NetworkPhase
  currentToken?: string
}

const phaseColor: Record<NetworkPhase, string> = {
  idle: "hsl(240, 5%, 35%)",
  drafting: "hsl(142, 71%, 45%)",
  verifying: "hsl(48, 96%, 53%)",
  correcting: "hsl(217, 91%, 60%)",
  complete: "hsl(142, 71%, 45%)",
}

const phaseLabel: Record<NetworkPhase, string> = {
  idle: "Idle",
  drafting: "Drafting",
  verifying: "Verifying",
  correcting: "Correcting",
  complete: "Complete",
}

/* Packet moving from edge -> cloud (draft tokens) */
function DraftPacket({ delay, phase }: { delay: number; phase: NetworkPhase }) {
  const active = phase === "drafting" || phase === "correcting"
  return (
    <motion.circle
      r="4"
      fill={phase === "correcting" ? "hsl(217, 91%, 60%)" : "hsl(142, 71%, 45%)"}
      initial={{ opacity: 0 }}
      animate={
        active
          ? { opacity: [0, 1, 1, 0], cx: [100, 340], cy: [90, 90] }
          : { opacity: 0, cx: 100, cy: 90 }
      }
      transition={{
        duration: 1.4,
        delay,
        repeat: active ? Infinity : 0,
        repeatDelay: 0.6,
        ease: "easeInOut",
      }}
    />
  )
}

/* Packet moving from cloud -> edge (verified / rejected) */
function VerifyPacket({ delay, phase }: { delay: number; phase: NetworkPhase }) {
  const active = phase === "verifying" || phase === "drafting"
  const color =
    phase === "verifying" ? "hsl(48, 96%, 53%)" : "hsl(217, 91%, 60%)"
  return (
    <motion.circle
      r="4"
      fill={color}
      initial={{ opacity: 0 }}
      animate={
        active
          ? { opacity: [0, 1, 1, 0], cx: [340, 100], cy: [110, 110] }
          : { opacity: 0, cx: 340, cy: 110 }
      }
      transition={{
        duration: 1.4,
        delay,
        repeat: active ? Infinity : 0,
        repeatDelay: 0.8,
        ease: "easeInOut",
      }}
    />
  )
}

export function NetworkVisualizer({ phase, currentToken }: NetworkVisualizerProps) {
  const edgeStroke = phase === "drafting" || phase === "correcting"
    ? "hsl(142, 71%, 45%)"
    : phase === "verifying"
      ? "hsl(48, 96%, 53%)"
      : phase === "complete"
        ? "hsl(142, 71%, 45%)"
        : "hsl(240, 5%, 25%)"

  const cloudStroke = phase === "verifying"
    ? "hsl(48, 96%, 53%)"
    : phase === "correcting"
      ? "hsl(217, 91%, 60%)"
      : phase === "drafting"
        ? "hsl(217, 91%, 60%)"
        : phase === "complete"
          ? "hsl(142, 71%, 45%)"
          : "hsl(240, 5%, 25%)"

  const lineOpacity = phase === "idle" ? 0.15 : 0.5

  return (
    <div className="flex items-center gap-4">
      <svg
        viewBox="0 0 440 200"
        className="h-28 w-full max-w-xl shrink-0"
        aria-label="Network diagram synchronized to chat"
      >
        {/* ── Edge Node ── */}
        <motion.rect
          x="10" y="50" width="130" height="100" rx="12"
          fill="hsl(240, 6%, 8%)"
          stroke={edgeStroke}
          strokeWidth="2"
          animate={{ stroke: edgeStroke }}
          transition={{ duration: 0.3 }}
        />
        {/* Edge glow */}
        <motion.rect
          x="10" y="50" width="130" height="100" rx="12"
          fill="none"
          stroke={edgeStroke}
          strokeWidth="6"
          strokeOpacity={phase !== "idle" ? 0.12 : 0}
          animate={{ strokeOpacity: phase !== "idle" ? [0.06, 0.15, 0.06] : 0 }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        <text x="75" y="85" textAnchor="middle" fill={edgeStroke} fontSize="11" fontWeight="700" fontFamily="var(--font-space-grotesk), sans-serif">
          Edge Draft
        </text>
        <text x="75" y="103" textAnchor="middle" fill="hsl(240, 5%, 45%)" fontSize="9" fontFamily="var(--font-jetbrains), monospace">
          RTX 3060
        </text>
        {/* Status dot */}
        <motion.circle
          cx="30" cy="66"
          r="4"
          fill={phaseColor[phase]}
          animate={phase !== "idle" && phase !== "complete" ? { r: [3.5, 5, 3.5] } : { r: 4 }}
          transition={{ duration: 1.2, repeat: Infinity }}
        />

        {/* ── Cloud Node ── */}
        <motion.rect
          x="300" y="50" width="130" height="100" rx="12"
          fill="hsl(240, 6%, 8%)"
          stroke={cloudStroke}
          strokeWidth="2"
          animate={{ stroke: cloudStroke }}
          transition={{ duration: 0.3 }}
        />
        {/* Cloud glow */}
        <motion.rect
          x="300" y="50" width="130" height="100" rx="12"
          fill="none"
          stroke={cloudStroke}
          strokeWidth="6"
          strokeOpacity={phase !== "idle" ? 0.12 : 0}
          animate={{ strokeOpacity: phase !== "idle" ? [0.06, 0.15, 0.06] : 0 }}
          transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
        />
        <text x="365" y="85" textAnchor="middle" fill={cloudStroke} fontSize="11" fontWeight="700" fontFamily="var(--font-space-grotesk), sans-serif">
          Cloud Target
        </text>
        <text x="365" y="103" textAnchor="middle" fill="hsl(240, 5%, 45%)" fontSize="9" fontFamily="var(--font-jetbrains), monospace">
          H100
        </text>
        {/* Status dot */}
        <motion.circle
          cx="320" cy="66"
          r="4"
          fill={phaseColor[phase]}
          animate={phase !== "idle" && phase !== "complete" ? { r: [3.5, 5, 3.5] } : { r: 4 }}
          transition={{ duration: 1.2, repeat: Infinity, delay: 0.3 }}
        />

        {/* ── Connection lines ── */}
        <motion.line
          x1="140" y1="90" x2="300" y2="90"
          stroke="hsl(142, 71%, 45%)"
          strokeWidth="1"
          strokeDasharray="6 4"
          animate={{ strokeOpacity: lineOpacity }}
          transition={{ duration: 0.3 }}
        />
        <motion.line
          x1="140" y1="110" x2="300" y2="110"
          stroke="hsl(217, 91%, 60%)"
          strokeWidth="1"
          strokeDasharray="6 4"
          animate={{ strokeOpacity: lineOpacity }}
          transition={{ duration: 0.3 }}
        />

        {/* Line labels */}
        <text x="220" y="84" textAnchor="middle" fill="hsl(240, 5%, 45%)" fontSize="8">Draft Tokens</text>
        <text x="220" y="126" textAnchor="middle" fill="hsl(240, 5%, 45%)" fontSize="8">Verified Tokens</text>

        {/* Arrowheads */}
        <polygon points="294,87 300,90 294,93" fill="hsl(142, 71%, 45%)" fillOpacity={lineOpacity} />
        <polygon points="146,107 140,110 146,113" fill="hsl(217, 91%, 60%)" fillOpacity={lineOpacity} />

        {/* ── Animated packets ── */}
        <DraftPacket delay={0} phase={phase} />
        <DraftPacket delay={0.7} phase={phase} />
        <DraftPacket delay={1.4} phase={phase} />
        <VerifyPacket delay={0.3} phase={phase} />
        <VerifyPacket delay={1.1} phase={phase} />

        {/* ── Phase label ── */}
        <motion.rect
          x="175" y="148" width="90" height="24" rx="6"
          fill="hsl(240, 6%, 10%)"
          stroke={phaseColor[phase]}
          strokeWidth="1"
          animate={{ stroke: phaseColor[phase] }}
          transition={{ duration: 0.3 }}
        />
        <motion.text
          x="220" y="164"
          textAnchor="middle"
          fontSize="10"
          fontWeight="600"
          fontFamily="var(--font-space-grotesk), sans-serif"
          animate={{ fill: phaseColor[phase] }}
          transition={{ duration: 0.3 }}
        >
          {phaseLabel[phase]}
        </motion.text>
      </svg>

      {/* Current token indicator */}
      <div className="flex shrink-0 flex-col gap-2">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <motion.span
            className="relative flex h-2 w-2"
            animate={phase !== "idle" && phase !== "complete" ? { scale: [1, 1.3, 1] } : {}}
            transition={{ duration: 1, repeat: Infinity }}
          >
            <span
              className="absolute inline-flex h-full w-full rounded-full opacity-75"
              style={{ backgroundColor: phaseColor[phase] }}
            />
            <span
              className="relative inline-flex h-2 w-2 rounded-full"
              style={{ backgroundColor: phaseColor[phase] }}
            />
          </motion.span>
          <span className="font-heading text-xs font-medium" style={{ color: phaseColor[phase] }}>
            {phaseLabel[phase]}
          </span>
        </div>
        {currentToken && phase !== "idle" && phase !== "complete" && (
          <motion.span
            key={currentToken}
            initial={{ opacity: 0, x: 4 }}
            animate={{ opacity: 1, x: 0 }}
            className="max-w-[120px] truncate rounded-md border border-border/30 bg-background/50 px-2 py-1 font-mono text-[11px] text-foreground/70"
          >
            {currentToken}
          </motion.span>
        )}
        <div className="flex flex-col gap-1 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-400" />
            Draft
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-yellow-400" />
            Verify
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-blue-400" />
            Correct
          </span>
        </div>
      </div>
    </div>
  )
}
