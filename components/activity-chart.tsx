"use client"

import { motion } from "framer-motion"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity } from "lucide-react"

const CHART_DATA = [
  { day: "Mon", tokens: 520000 },
  { day: "Tue", tokens: 680000 },
  { day: "Wed", tokens: 590000 },
  { day: "Thu", tokens: 740000 },
  { day: "Fri", tokens: 620000 },
  { day: "Sat", tokens: 480000 },
  { day: "Sun", tokens: 570000 },
]

const GREEN = "hsl(142, 71%, 45%)"

function formatTokens(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`
  return String(value)
}

interface TooltipPayloadItem {
  value: number
  dataKey: string
}

interface CustomTooltipProps {
  active?: boolean
  payload?: TooltipPayloadItem[]
  label?: string
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-border/60 bg-card px-3 py-2 shadow-xl">
      <p className="text-[10px] font-medium text-muted-foreground">{label}</p>
      <p className="font-mono text-sm font-bold text-foreground">
        {formatTokens(payload[0].value)} tokens
      </p>
    </div>
  )
}

export function ActivityChart() {
  return (
    <motion.div
      className="flex flex-col"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3, duration: 0.4 }}
    >
      <Card className="flex flex-1 flex-col border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 font-heading text-sm font-medium text-muted-foreground">
            <Activity className="h-4 w-4 text-primary" />
            Tokens Drafted — Last 7 Days
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 pb-4 pr-2">
          <div className="h-full min-h-[13rem]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={CHART_DATA}
                margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(240, 4%, 16%)"
                  vertical={false}
                />
                <XAxis
                  dataKey="day"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(240, 5%, 55%)" }}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={formatTokens}
                  tick={{ fontSize: 11, fill: "hsl(240, 5%, 55%)" }}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: "hsl(240, 4%, 16%)", opacity: 0.5 }} />
                <Bar
                  dataKey="tokens"
                  fill={GREEN}
                  radius={[4, 4, 0, 0]}
                  style={{ filter: "none" }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
