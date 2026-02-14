"use client"

import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Wallet } from "lucide-react"
import { useEffect, useState } from "react"

interface Payout {
  date: string
  amount: number
  status: "Completed" | "Pending"
}

export function RecentPayoutsTable() {
  const [payouts, setPayouts] = useState<Payout[]>([])

  useEffect(() => {
    // Fetch payouts on mount
    fetch("http://localhost:8000/api/provider/payouts?limit=6")
      .then((res) => res.json())
      .then((data) => setPayouts(data.payouts))
      .catch((err) => console.error("Failed to fetch payouts:", err))

    // Poll for updates every 10 seconds
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/provider/payouts?limit=6")
        .then((res) => res.json())
        .then((data) => setPayouts(data.payouts))
        .catch((err) => console.error("Failed to fetch payouts:", err))
    }, 10000)

    return () => clearInterval(interval)
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4, duration: 0.4 }}
    >
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 font-heading text-sm font-medium text-muted-foreground">
            <Wallet className="h-4 w-4 text-primary" />
            Recent Payouts
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4">
          <div className="overflow-hidden rounded-lg border border-border/40">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/40 bg-secondary/30">
                  <th className="px-3 py-2 text-left font-medium text-muted-foreground">
                    Date
                  </th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">
                    Amount
                  </th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {payouts.length === 0 ? (
                  <tr>
                    <td
                      colSpan={3}
                      className="px-3 py-6 text-center text-muted-foreground"
                    >
                      No payouts yet
                    </td>
                  </tr>
                ) : (
                  payouts.map((payout) => (
                    <tr
                      key={`${payout.date}-${payout.amount}`}
                      className="border-b border-border/20 transition-colors hover:bg-secondary/20"
                    >
                      <td className="px-3 py-2.5 text-foreground">{payout.date}</td>
                      <td className="px-3 py-2.5 text-right font-mono font-semibold text-foreground">
                        ${payout.amount.toFixed(2)}
                      </td>
                      <td className="px-3 py-2.5 text-right">
                        {payout.status === "Completed" ? (
                          <Badge
                            variant="outline"
                            className="border-primary/30 bg-primary/10 text-primary text-[10px]"
                          >
                            Completed
                          </Badge>
                        ) : (
                          <Badge
                            variant="outline"
                            className="border-yellow-500/30 bg-yellow-500/10 text-yellow-400 text-[10px]"
                          >
                            Pending
                          </Badge>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
