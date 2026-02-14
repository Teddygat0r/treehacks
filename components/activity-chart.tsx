"use client"

import { motion } from "framer-motion"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity } from "lucide-react"
import { useEffect, useState } from "react"

interface ActivityDataPoint {
  time: string
  tokens: number
  earnings: number
  requests: number
}

const EMPTY_CHART_DATA = [
  { time: "00:00", tokens: 0 },
  { time: "04:00", tokens: 0 },
  { time: "08:00", tokens: 0 },
  { time: "12:00", tokens: 0 },
  { time: "16:00", tokens: 0 },
  { time: "20:00", tokens: 0 },
]

const GREEN = "hsl(142, 71%, 45%)"
const GREEN_FILL = "hsl(142, 71%, 45%)"

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
  const [activityData, setActivityData] = useState<ActivityDataPoint[]>([])

  useEffect(() => {
    // Fetch activity on mount
    fetch("http://localhost:8000/api/provider/activity?hours=24")
      .then((res) => res.json())
      .then((data) => setActivityData(data.activity))
      .catch((err) => console.error("Failed to fetch activity:", err))

    // Poll for updates every 10 seconds
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/provider/activity?hours=24")
        .then((res) => res.json())
        .then((data) => setActivityData(data.activity))
        .catch((err) => console.error("Failed to fetch activity:", err))
    }, 10000)

    return () => clearInterval(interval)
  }, [])

  const chartData = activityData.length > 0 ? activityData : EMPTY_CHART_DATA

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3, duration: 0.4 }}
    >
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 font-heading text-sm font-medium text-muted-foreground">
            <Activity className="h-4 w-4 text-primary" />
            Tokens Drafted — Last 24 Hours
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-4 pr-2">
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                margin={{ top: 8, right: 8, left: -16, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="tokenGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={GREEN_FILL} stopOpacity={0.25} />
                    <stop offset="100%" stopColor={GREEN_FILL} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(240, 4%, 16%)"
                  vertical={false}
                />
                <XAxis
                  dataKey="time"
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
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="tokens"
                  stroke={GREEN}
                  strokeWidth={2}
                  fill="url(#tokenGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
