"use client"

import { useEffect, useState } from "react"
import { Server, Zap, TrendingDown } from "lucide-react"
import { NetworkStatCard } from "@/components/network-stat-card"
import { fetchStats, fetchModelPairs } from "@/lib/api"
import type { NetworkStats, ModelPair } from "@/lib/types"

export function MarketplaceStats() {
  const [stats, setStats] = useState<NetworkStats>()
  const [modelCount, setModelCount] = useState<number>(0)

  useEffect(() => {
    const loadStats = async () => {
      try {
        const [networkStats, modelPairs] = await Promise.all([
          fetchStats(),
          fetchModelPairs(),
        ])
        setStats(networkStats)
        setModelCount(modelPairs.filter((p) => p.available).length)
      } catch (error) {
        console.error("Failed to load marketplace stats:", error)
      }
    }

    loadStats()
    const interval = setInterval(loadStats, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  if (!stats) {
    return (
      <div className="grid gap-4 md:grid-cols-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-24 animate-pulse rounded-lg bg-card/50" />
        ))}
      </div>
    )
  }

  const totalProviders = stats.active_draft_nodes + stats.active_target_nodes
  const avgCostSavings = 45 // Mock: 45% average cost savings

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <NetworkStatCard
        icon={Server}
        label="Active Providers"
        value={totalProviders.toString()}
        sub="nodes online"
        color="#22c55e"
        index={0}
        breakdown={{
          draft: stats.active_draft_nodes,
          target: stats.active_target_nodes,
        }}
      />
      <NetworkStatCard
        icon={Zap}
        label="Available Models"
        value={modelCount.toString()}
        sub="model pairs"
        color="#3b82f6"
        index={1}
      />
      <NetworkStatCard
        icon={TrendingDown}
        label="Cost Savings"
        value={`${avgCostSavings}%`}
        sub="vs traditional"
        color="#22c55e"
        index={2}
      />
    </div>
  )
}
