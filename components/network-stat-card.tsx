"use client"

import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import type { LucideIcon } from "lucide-react"

interface NetworkStatCardProps {
  icon: LucideIcon
  label: string
  value: string
  sub: string
  color: string
  index: number
  /** Optional breakdown shown under the main value (e.g. Draft / Target counts) */
  breakdown?: { draft: number; target: number }
}

const cardVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.4, ease: "easeOut" as const },
  }),
}

export function NetworkStatCard({ icon: Icon, label, value, sub, color, index, breakdown }: NetworkStatCardProps) {
  return (
    <motion.div custom={index} initial="hidden" animate="visible" variants={cardVariants}>
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardContent className="p-5">
          <div className="flex items-center gap-4">
            <div
              className="flex h-12 w-12 min-h-12 min-w-12 shrink-0 items-center justify-center rounded-lg [color:var(--icon-color)]"
              style={{
                "--icon-color": color,
                backgroundColor: `${color}15`,
                boxShadow: `inset 0 0 0 1px ${color}30`,
              } as React.CSSProperties}
            >
              <Icon
                className="shrink-0"
                size={24}
                color={color}
                strokeWidth={2}
              />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs text-muted-foreground">{label}</p>
              <div className="flex items-baseline gap-1.5">
                <span className="font-heading text-2xl font-bold tracking-tight text-foreground">{value}</span>
                <span className="text-xs text-muted-foreground">{sub}</span>
              </div>
            </div>
            {breakdown != null && (
              <div className="flex shrink-0 flex-col gap-1.5 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 shrink-0 rounded-full bg-green-500" />
                  <span className="w-14 shrink-0">Draft:</span>
                  <span className="font-medium tabular-nums text-foreground">{breakdown.draft}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 shrink-0 rounded-full bg-blue-500" />
                  <span className="w-14 shrink-0">Target:</span>
                  <span className="font-medium tabular-nums text-foreground">{breakdown.target}</span>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
