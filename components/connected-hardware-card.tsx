"use client"

import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Cpu, Info } from "lucide-react"
import { useEffect, useState } from "react"

export function ConnectedHardwareCard() {
  const [stats, setStats] = useState({
    total_tokens: 0,
    total_inferences: 0,
    average_acceptance_rate: 0,
  })

  useEffect(() => {
    // Fetch stats on mount
    fetch("http://localhost:8000/api/provider/earnings")
      .then((res) => res.json())
      .then((data) => {
        setStats({
          total_tokens: data.total_tokens || 0,
          total_inferences: data.total_inferences || 0,
          average_acceptance_rate: data.average_acceptance_rate || 0,
        })
      })
      .catch((err) => console.error("Failed to fetch stats:", err))

    // Poll for updates every 5 seconds
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/provider/earnings")
        .then((res) => res.json())
        .then((data) => {
          setStats({
            total_tokens: data.total_tokens || 0,
            total_inferences: data.total_inferences || 0,
            average_acceptance_rate: data.average_acceptance_rate || 0,
          })
        })
        .catch((err) => console.error("Failed to fetch stats:", err))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  function formatTokens(value: number) {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
    if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
    return String(value)
  }

  const acceptanceRate = Math.round(stats.average_acceptance_rate)

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.4 }}
    >
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <CardTitle className="font-heading text-sm font-medium text-muted-foreground">
            Connected Hardware
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Node Identity */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 ring-1 ring-accent/20">
                <Cpu className="h-5 w-5 text-accent" />
              </div>
              <div>
                <p className="font-heading text-sm font-semibold text-foreground">
                  Modal-Draft-Node
                </p>
                <p className="text-xs text-muted-foreground">Qwen 1.5B on A10G GPU</p>
              </div>
            </div>
            <Badge
              variant="outline"
              className="gap-1.5 border-primary/30 bg-primary/10 text-primary"
            >
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
              </span>
              Actively Drafting
            </Badge>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 gap-4 rounded-lg border border-border/40 bg-secondary/20 p-4">
            <div>
              <p className="text-xs text-muted-foreground">Total Tokens Drafted</p>
              <p className="mt-1 font-heading text-lg font-bold text-foreground">
                {formatTokens(stats.total_tokens)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Total Inferences</p>
              <p className="mt-1 font-heading text-lg font-bold text-foreground">
                {stats.total_inferences}
              </p>
            </div>
          </div>

          {/* Draft Acceptance Rate */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <p className="font-heading text-sm font-medium text-foreground">
                Draft Acceptance Rate
              </p>
              <span className="font-mono text-sm font-bold text-foreground">
                {acceptanceRate}%
              </span>
            </div>
            <Progress value={acceptanceRate} className="h-2.5" />
            <div className="flex items-start gap-1.5 rounded-md border border-border/30 bg-secondary/30 px-3 py-2">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                Higher acceptance rates earn higher network rewards. Improve your rate
                by ensuring low latency and keeping your node online during peak hours.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
