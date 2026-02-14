"use client"

import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { DollarSign, TrendingUp, TrendingDown } from "lucide-react"
import { useEffect, useState } from "react"

export function TotalEarningsCard() {
  const [earnings, setEarnings] = useState({
    total_earnings: 0,
    today_earnings: 0,
    change_percentage: 0,
    total_inferences: 0,
  })

  useEffect(() => {
    // Fetch earnings on mount
    fetch("http://localhost:8000/api/provider/earnings")
      .then((res) => res.json())
      .then((data) => setEarnings(data))
      .catch((err) => console.error("Failed to fetch earnings:", err))

    // Poll for updates every 5 seconds
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/provider/earnings")
        .then((res) => res.json())
        .then((data) => setEarnings(data))
        .catch((err) => console.error("Failed to fetch earnings:", err))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const isPositive = earnings.change_percentage >= 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1, duration: 0.4 }}
    >
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardContent className="p-6">
          <div className="flex items-start">
            <div>
              <div className="flex items-center gap-2">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 ring-1 ring-primary/20">
                  <DollarSign className="h-4 w-4 text-primary" />
                </div>
                <p className="font-heading text-sm font-medium text-muted-foreground">
                  Total Earnings ({earnings.total_inferences} inferences)
                </p>
              </div>
              <div className="mt-4 flex items-baseline gap-3">
                <span className="font-heading text-5xl font-bold tracking-tight text-foreground">
                  ${earnings.total_earnings.toFixed(2)}
                </span>
                {earnings.change_percentage !== 0 && (
                  <span
                    className={`flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold ${
                      isPositive
                        ? "bg-primary/10 text-primary"
                        : "bg-destructive/10 text-destructive"
                    }`}
                  >
                    {isPositive ? (
                      <TrendingUp className="h-3 w-3" />
                    ) : (
                      <TrendingDown className="h-3 w-3" />
                    )}
                    {isPositive ? "+" : ""}
                    {earnings.change_percentage.toFixed(1)}% vs yesterday
                  </span>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
