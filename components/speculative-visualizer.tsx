"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Play, RotateCcw, Cpu, Cloud, Check, X, Zap } from "lucide-react"
import { cn } from "@/lib/utils"
import logData from "@/data/speculative-log.json"

/* ── Constants ── */
const GREEN = "hsl(142, 71%, 45%)"
const RED = "hsl(0, 72%, 51%)"
const BLUE = "hsl(217, 91%, 60%)"
const YELLOW = "hsl(48, 96%, 53%)"
const DIM = "hsl(240, 5%, 25%)"

/* ── Types ── */
type VisPhase = "idle" | "running" | "complete"

interface AnimToken {
  id: number
  text: string
  type: "accepted" | "rejected" | "corrected"
  stage: "draft" | "fly-right" | "verify" | "fly-result" | "done"
  roundNum: number
}

interface RoundStats {
  round_num: number
  drafted: number
  accepted: number
  corrected: number
  acceptance_rate: number
  verification_time_ms: number
}

interface LogToken {
  text: string
  type: "accepted" | "rejected" | "corrected"
}

interface LogRound {
  round_num: number
  drafted: number
  accepted: number
  corrected: number
  verification_time_ms: number
  acceptance_rate: number
  tokens: LogToken[]
}

interface LogData {
  prompt: string
  rounds: LogRound[]
  summary: {
    total_tokens: number
    draft_tokens_generated: number
    draft_tokens_accepted: number
    acceptance_rate: number
    speculation_rounds: number
    generation_time_ms: number
  }
}

/* ── Sub-components ── */

function NodePanel({
  label,
  sublabel,
  icon: Icon,
  active,
  color,
  children,
}: {
  label: string
  sublabel: string
  icon: React.ElementType
  active: boolean
  color: string
  children?: React.ReactNode
}) {
  return (
    <motion.div
      className={cn(
        "relative flex w-56 shrink-0 flex-col rounded-2xl border-2 p-4",
        "bg-card/60 backdrop-blur-sm",
      )}
      style={{ borderColor: active ? color : DIM }}
      animate={{ borderColor: active ? color : DIM }}
      transition={{ duration: 0.3 }}
    >
      {active && (
        <motion.div
          className="pointer-events-none absolute inset-0 rounded-2xl"
          style={{ boxShadow: `0 0 30px 6px ${color}` }}
          animate={{ opacity: [0.03, 0.12, 0.03] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      )}
      <div className="mb-3 flex items-center gap-2">
        <div
          className="flex h-8 w-8 items-center justify-center rounded-lg"
          style={{ backgroundColor: `${color}20` }}
        >
          <Icon className="h-4 w-4" style={{ color }} />
        </div>
        <div>
          <div className="font-heading text-xs font-bold" style={{ color }}>
            {label}
          </div>
          <div className="font-mono text-[10px] text-muted-foreground">{sublabel}</div>
        </div>
        {active && (
          <motion.div
            className="ml-auto h-2 w-2 rounded-full"
            style={{ backgroundColor: color }}
            animate={{ scale: [1, 1.5, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
        )}
      </div>
      <div className="min-h-[120px]">{children}</div>
    </motion.div>
  )
}

function TokenPill({
  text,
  type,
  animateIn,
  showIcon,
  neutral,
}: {
  text: string
  type: "accepted" | "rejected" | "corrected"
  animateIn?: boolean
  showIcon?: boolean
  neutral?: boolean
}) {
  const bg = neutral
    ? "bg-green-500/10 border-green-500/25 text-green-200"
    : type === "accepted"
      ? "bg-green-500/20 border-green-500/40 text-green-300"
      : type === "rejected"
        ? "bg-red-500/20 border-red-500/40 text-red-300"
        : "bg-blue-500/20 border-blue-500/40 text-blue-300"

  const icon =
    type === "accepted" ? (
      <Check className="h-2.5 w-2.5 text-green-400" />
    ) : type === "rejected" ? (
      <X className="h-2.5 w-2.5 text-red-400" />
    ) : (
      <Zap className="h-2.5 w-2.5 text-blue-400" />
    )

  return (
    <motion.span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono text-[11px]",
        bg,
      )}
      initial={animateIn ? { opacity: 0, scale: 0.5, y: 8 } : false}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.25, type: "spring", stiffness: 400 }}
    >
      {showIcon && icon}
      {text}
    </motion.span>
  )
}

function FlyingToken({
  text,
  type,
  direction,
  onDone,
  neutral,
}: {
  text: string
  type: "accepted" | "rejected" | "corrected"
  direction: "right" | "left"
  onDone: () => void
  neutral?: boolean
}) {
  const color = neutral ? GREEN : type === "accepted" ? GREEN : type === "rejected" ? RED : BLUE

  return (
    <motion.div
      className="absolute top-1/2 flex -translate-y-1/2 items-center gap-1 rounded-md border px-2 py-1 font-mono text-[11px]"
      style={{
        borderColor: `${color}66`,
        backgroundColor: `${color}15`,
        color,
        boxShadow: `0 0 12px 2px ${color}44`,
      }}
      initial={{
        left: direction === "right" ? "0%" : "100%",
        opacity: 0,
        scale: 0.6,
      }}
      animate={{
        left: direction === "right" ? "100%" : "0%",
        opacity: [0, 1, 1, 0.6],
        scale: [0.6, 1.1, 1, 0.9],
      }}
      transition={{ duration: 0.5, ease: "easeInOut" }}
      onAnimationComplete={onDone}
    >
      {text}
    </motion.div>
  )
}

function StatsBar({ rounds }: { rounds: RoundStats[] }) {
  const latest = rounds[rounds.length - 1]
  if (!latest) return null

  const totalDrafted = rounds.reduce((s, r) => s + r.drafted, 0)
  const totalAccepted = rounds.reduce((s, r) => s + r.accepted, 0)
  const overallRate = totalDrafted > 0 ? (totalAccepted / totalDrafted) * 100 : 0

  return (
    <motion.div
      className="flex flex-wrap items-center justify-center gap-4 rounded-xl border border-border/50 bg-card/40 px-4 py-2 text-xs backdrop-blur-sm"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground">Round</span>
        <span className="font-mono font-bold text-foreground">{latest.round_num}</span>
      </div>
      <div className="h-3 w-px bg-border/50" />
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground">Drafted</span>
        <span className="font-mono font-bold text-foreground">{totalDrafted}</span>
      </div>
      <div className="h-3 w-px bg-border/50" />
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground">Accepted</span>
        <span className="font-mono font-bold text-green-400">{totalAccepted}</span>
      </div>
      <div className="h-3 w-px bg-border/50" />
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground">Rate</span>
        <span className="font-mono font-bold text-yellow-400">{overallRate.toFixed(0)}%</span>
      </div>
      <div className="h-3 w-px bg-border/50" />
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground">Verify</span>
        <span className="font-mono font-bold text-blue-400">
          {latest.verification_time_ms.toFixed(0)}ms
        </span>
      </div>
    </motion.div>
  )
}

/* ── Main export ── */
export function SpeculativeVisualizer() {
  const data = logData as LogData

  const [phase, setPhase] = useState<VisPhase>("idle")
  const [tokens, setTokens] = useState<AnimToken[]>([])
  const [rounds, setRounds] = useState<RoundStats[]>([])
  const [resultTokens, setResultTokens] = useState<{ text: string; type: string }[]>([])
  const [finalStats, setFinalStats] = useState<LogData["summary"] | null>(null)

  const tokenIdRef = useRef(0)
  const cancelledRef = useRef(false)
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([])

  const reset = useCallback(() => {
    cancelledRef.current = true
    timeoutsRef.current.forEach(clearTimeout)
    timeoutsRef.current = []
    setPhase("idle")
    setTokens([])
    setRounds([])
    setResultTokens([])
    setFinalStats(null)
    tokenIdRef.current = 0
  }, [])

  const schedule = useCallback((fn: () => void, delay: number) => {
    const id = setTimeout(() => {
      if (!cancelledRef.current) fn()
    }, delay)
    timeoutsRef.current.push(id)
  }, [])

  const startVisualization = useCallback(() => {
    reset()
    // Allow a tick for reset to clear
    setTimeout(() => {
      cancelledRef.current = false
      setPhase("running")

      let globalDelay = 0
      const DRAFT_STAGGER = 100   // stagger tokens appearing in draft node
      const BATCH_HOLD = 400      // pause after all drafted before flying
      const FLY_DURATION = 500    // time for flight animation
      const VERIFY_HOLD = 300     // pause after last verdict before clearing
      const ROUND_GAP = 600       // pause between rounds

      data.rounds.forEach((round) => {
        const roundTokens = round.tokens
        const batchIds: number[] = []

        // Stage 1: All tokens appear in draft node (staggered slightly)
        roundTokens.forEach((tok, i) => {
          const id = ++tokenIdRef.current
          batchIds.push(id)
          const animToken: AnimToken = {
            id,
            text: tok.text,
            type: tok.type,
            stage: "draft",
            roundNum: round.round_num,
          }
          schedule(() => {
            setTokens((prev) => [...prev, animToken])
          }, globalDelay + i * DRAFT_STAGGER)
        })

        const draftDone = globalDelay + roundTokens.length * DRAFT_STAGGER + BATCH_HOLD

        // Stage 2: Entire batch flies to target at once
        schedule(() => {
          setTokens((prev) =>
            prev.map((t) =>
              batchIds.includes(t.id) ? { ...t, stage: "fly-right" } : t,
            ),
          )
        }, draftDone)

        const flyDone = draftDone + FLY_DURATION

        // Stage 3: Accepted tokens appear all at once; corrected tokens appear after a delay
        const acceptedIds: number[] = []
        const correctedIds: number[] = []
        roundTokens.forEach((tok, i) => {
          if (tok.type === "accepted") acceptedIds.push(batchIds[i])
          else if (tok.type === "rejected") acceptedIds.push(batchIds[i]) // rejected shows instantly too
          else if (tok.type === "corrected") correctedIds.push(batchIds[i])
        })

        // All accepted + rejected appear at once (parallel verification)
        schedule(() => {
          setTokens((prev) =>
            prev.map((t) =>
              acceptedIds.includes(t.id) ? { ...t, stage: "verify" } : t,
            ),
          )
        }, flyDone)

        // Corrected tokens appear after a delay (autoregressive generation)
        const CORRECTION_DELAY = 600
        correctedIds.forEach((id, i) => {
          schedule(() => {
            setTokens((prev) =>
              prev.map((t) => (t.id === id ? { ...t, stage: "verify" } : t)),
            )
          }, flyDone + CORRECTION_DELAY + i * 300)
        })

        const lastCorrectionTime = correctedIds.length > 0
          ? CORRECTION_DELAY + correctedIds.length * 300
          : 0
        const verifyDone = flyDone + Math.max(VERIFY_HOLD, lastCorrectionTime + VERIFY_HOLD)

        // Stage 4: Clear batch — move accepted/corrected to result
        schedule(() => {
          setTokens((prev) =>
            prev.map((t) =>
              batchIds.includes(t.id) ? { ...t, stage: "done" } : t,
            ),
          )
          roundTokens.forEach((tok) => {
            if (tok.type === "accepted" || tok.type === "corrected") {
              setResultTokens((prev) => [...prev, { text: tok.text, type: tok.type }])
            }
          })
        }, verifyDone)

        // Emit round stats
        schedule(() => {
          setRounds((prev) => [
            ...prev,
            {
              round_num: round.round_num,
              drafted: round.drafted,
              accepted: round.accepted,
              corrected: round.corrected,
              acceptance_rate: round.acceptance_rate,
              verification_time_ms: round.verification_time_ms,
            },
          ])
        }, verifyDone)

        globalDelay = verifyDone + ROUND_GAP
      })

      // After all rounds, show final stats
      schedule(() => {
        setFinalStats(data.summary)
        setPhase("complete")
      }, globalDelay + 1500)
    }, 50)
  }, [reset, data, schedule])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelledRef.current = true
      timeoutsRef.current.forEach(clearTimeout)
    }
  }, [])

  // Separate tokens by stage for rendering
  const draftStageTokens = tokens.filter((t) => t.stage === "draft")
  const flyingTokens = tokens.filter((t) => t.stage === "fly-right")
  const verifyTokens = tokens.filter((t) => t.stage === "verify")

  const isDraftActive = phase === "running" && (draftStageTokens.length > 0 || flyingTokens.length > 0)
  const isTargetActive = phase === "running" && (flyingTokens.length > 0 || verifyTokens.length > 0)

  return (
    <div className="flex flex-1 flex-col items-center gap-6 overflow-y-auto px-6 py-8">
      {/* Title */}
      <div className="text-center">
        <h2 className="font-heading text-xl font-bold tracking-tight text-foreground">
          Speculative Decoding Visualizer
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Watch how draft and target models collaborate to generate text faster
        </p>
      </div>

      {/* Fixed prompt display */}
      <div className="w-full max-w-2xl rounded-xl border border-border/50 bg-card/40 px-4 py-3 backdrop-blur-sm">
        <div className="mb-1 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          Prompt
        </div>
        <div className="font-mono text-sm text-foreground">{data.prompt}</div>
      </div>

      {/* Main visualization area */}
      <div className="flex w-full max-w-4xl items-stretch gap-0">
        {/* Draft node */}
        <NodePanel
          label="Draft Node"
          sublabel="Qwen 1.5B · RTX 3060"
          icon={Cpu}
          active={isDraftActive}
          color={GREEN}
        >
          <div className="flex flex-wrap gap-1">
            <AnimatePresence>
              {draftStageTokens.map((t) => (
                <motion.div
                  key={t.id}
                  exit={{ opacity: 0, scale: 0.5, x: 20 }}
                  transition={{ duration: 0.2 }}
                >
                  <TokenPill text={t.text} type={t.type} animateIn neutral />
                </motion.div>
              ))}
            </AnimatePresence>
            {phase === "running" && draftStageTokens.length === 0 && flyingTokens.length === 0 && (
              <span className="text-[10px] text-muted-foreground/60">Generating tokens...</span>
            )}
          </div>
        </NodePanel>

        {/* Connection lane */}
        <div className="relative flex flex-1 flex-col justify-center px-2">
          {/* Top label */}
          <div className="absolute -top-4 left-1/2 -translate-x-1/2 whitespace-nowrap text-[9px] font-medium uppercase tracking-widest text-muted-foreground/50">
            Token Transfer
          </div>

          {/* Draft → Target lane */}
          <div className="relative mb-4 flex items-center">
            <span className="absolute -top-3 left-1/2 -translate-x-1/2 text-[9px] text-green-400/60">
              Draft Tokens →
            </span>
            <motion.div
              className="h-px w-full"
              style={{
                backgroundImage: `repeating-linear-gradient(to right, ${GREEN} 0px, ${GREEN} 6px, transparent 6px, transparent 12px)`,
              }}
              animate={{ opacity: isDraftActive ? 0.5 : 0.12 }}
              transition={{ duration: 0.3 }}
            />
            <div
              className="ml-[-6px] h-0 w-0 shrink-0 border-y-[4px] border-l-[6px] border-y-transparent"
              style={{ borderLeftColor: GREEN, opacity: isDraftActive ? 0.6 : 0.15 }}
            />
            {/* Flying tokens */}
            <div className="absolute inset-0">
              <AnimatePresence>
                {flyingTokens.map((t) => (
                  <FlyingToken
                    key={t.id}
                    text={t.text}
                    type={t.type}
                    direction="right"
                    onDone={() => {}}
                    neutral
                  />
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* Target → Result lane */}
          <div className="relative flex items-center">
            <span className="absolute -bottom-3 left-1/2 -translate-x-1/2 text-[9px] text-blue-400/60">
              ← Verified
            </span>
            <div
              className="mr-[-6px] h-0 w-0 shrink-0 border-y-[4px] border-r-[6px] border-y-transparent"
              style={{ borderRightColor: BLUE, opacity: isTargetActive ? 0.6 : 0.15 }}
            />
            <motion.div
              className="h-px w-full"
              style={{
                backgroundImage: `repeating-linear-gradient(to right, ${BLUE} 0px, ${BLUE} 6px, transparent 6px, transparent 12px)`,
              }}
              animate={{ opacity: isTargetActive ? 0.5 : 0.12 }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        {/* Target node */}
        <NodePanel
          label="Target Node"
          sublabel="Qwen 7B · H100"
          icon={Cloud}
          active={isTargetActive}
          color={YELLOW}
        >
          <div className="flex flex-wrap gap-1">
            <AnimatePresence>
              {verifyTokens.map((t) => (
                <motion.div
                  key={t.id}
                  initial={{ opacity: 0, scale: 0.8, x: -10 }}
                  animate={{ opacity: 1, scale: 1, x: 0 }}
                  exit={{ opacity: 0, scale: 0.5, y: 10 }}
                  transition={{ duration: 0.25 }}
                >
                  <TokenPill text={t.text} type={t.type} showIcon animateIn />
                </motion.div>
              ))}
            </AnimatePresence>
            {phase === "running" && verifyTokens.length === 0 && !isDraftActive && (
              <span className="text-[10px] text-muted-foreground/60">Waiting for tokens...</span>
            )}
          </div>
        </NodePanel>
      </div>

      {/* Stats bar */}
      {rounds.length > 0 && <StatsBar rounds={rounds} />}

      {/* Result text */}
      {resultTokens.length > 0 && (
        <motion.div
          className="w-full max-w-2xl rounded-xl border border-border/50 bg-card/40 px-4 py-3 backdrop-blur-sm"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="mb-2 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
            Generated Output
          </div>
          <div className="flex flex-wrap gap-0.5 font-mono text-sm leading-relaxed">
            {resultTokens.map((t, i) => (
              <span
                key={i}
                className={cn(
                  "transition-colors",
                  t.type === "accepted"
                    ? "text-green-300"
                    : t.type === "corrected"
                      ? "text-blue-300"
                      : "text-foreground",
                )}
              >
                {t.text}
              </span>
            ))}
            {phase === "running" && (
              <motion.span
                className="inline-block h-4 w-1.5 rounded-sm bg-foreground/60"
                animate={{ opacity: [1, 0, 1] }}
                transition={{ duration: 0.8, repeat: Infinity }}
              />
            )}
          </div>
        </motion.div>
      )}

      {/* Final stats */}
      {phase === "complete" && finalStats && (
        <motion.div
          className="w-full max-w-2xl rounded-xl border border-green-500/30 bg-green-500/5 px-4 py-3"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="mb-2 text-[10px] font-medium uppercase tracking-widest text-green-400">
            Complete
          </div>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="font-mono text-lg font-bold text-foreground">
                {finalStats.total_tokens}
              </div>
              <div className="text-[10px] text-muted-foreground">Total Tokens</div>
            </div>
            <div>
              <div className="font-mono text-lg font-bold text-green-400">
                {(finalStats.acceptance_rate * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-muted-foreground">Acceptance Rate</div>
            </div>
            <div>
              <div className="font-mono text-lg font-bold text-blue-400">
                {finalStats.speculation_rounds}
              </div>
              <div className="text-[10px] text-muted-foreground">Rounds</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Start / Restart button */}
      <motion.button
        className={cn(
          "flex items-center gap-2 rounded-xl px-6 py-3 font-heading text-sm font-bold tracking-wide transition-colors",
          phase === "running"
            ? "cursor-not-allowed border border-border/50 bg-card/40 text-muted-foreground"
            : "bg-primary text-primary-foreground hover:bg-primary/90",
        )}
        onClick={phase === "running" ? undefined : startVisualization}
        disabled={phase === "running"}
        whileHover={phase !== "running" ? { scale: 1.03 } : {}}
        whileTap={phase !== "running" ? { scale: 0.97 } : {}}
      >
        {phase === "idle" && (
          <>
            <Play className="h-4 w-4" />
            Start Visualization
          </>
        )}
        {phase === "running" && (
          <>
            <motion.div
              className="h-4 w-4 rounded-full border-2 border-current border-t-transparent"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
            Running...
          </>
        )}
        {phase === "complete" && (
          <>
            <RotateCcw className="h-4 w-4" />
            Restart
          </>
        )}
      </motion.button>
    </div>
  )
}
