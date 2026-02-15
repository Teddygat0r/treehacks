"use client"

import { motion } from "framer-motion"
import { ArrowRight, Zap } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { ModelPair } from "@/lib/types"

interface ModelPairCardProps {
  pair: ModelPair
  index: number
}

const cardVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.4, ease: "easeOut" as const },
  }),
}

export function ModelPairCard({ pair, index }: ModelPairCardProps) {
  return (
    <motion.div
      custom={index}
      initial="hidden"
      animate="visible"
      variants={cardVariants}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.2 }}
    >
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm h-full">
        <CardContent className="p-5">
          <div className="mb-4 flex items-start justify-between">
            <div className="flex items-center gap-2">
              <div
                className="flex h-8 w-8 items-center justify-center rounded-md"
                style={{
                  backgroundColor: "#22c55e15",
                  boxShadow: "inset 0 0 0 1px #22c55e30",
                }}
              >
                <Zap size={16} color="#22c55e" strokeWidth={2} />
              </div>
              <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                {pair.category}
              </span>
            </div>
            <Badge
              variant="outline"
              className="border-green-500/30 bg-green-500/10 text-green-500"
            >
              {Math.round(pair.acceptance_rate * 100)}% accept
            </Badge>
          </div>

          <div className="mb-4 space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <span className="font-mono font-medium text-foreground">
                {pair.draft_model}
              </span>
              <ArrowRight size={16} className="text-muted-foreground" />
              <span className="font-mono font-medium text-foreground">
                {pair.target_model}
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <span>
                {pair.draft_size} → {pair.target_size}
              </span>
              <span className="h-1 w-1 rounded-full bg-muted-foreground/50" />
              <span className="font-semibold text-blue-500">
                {pair.speedup} faster
              </span>
            </div>
          </div>

          <div className="mb-4 rounded-md border border-border/50 bg-background/50 p-3">
            <div className="text-xs text-muted-foreground">Price</div>
            <div className="flex items-baseline gap-1">
              <span className="font-heading text-xl font-bold text-foreground">
                ${pair.price_per_1m.toFixed(2)}
              </span>
              <span className="text-xs text-muted-foreground">/ 1M tokens</span>
            </div>
          </div>

          <button
            className="w-full rounded-md bg-green-500 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-green-600 disabled:opacity-50"
            disabled={!pair.available}
          >
            {pair.available ? "Try Now" : "Coming Soon"}
          </button>
        </CardContent>
      </Card>
    </motion.div>
  )
}
