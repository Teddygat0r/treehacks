"use client"

import { useState, useEffect } from "react"
import { DraftNodesTable } from "@/components/draft-nodes-table"
import { TargetNodesTable } from "@/components/target-nodes-table"
import { NetworkStatCard } from "@/components/network-stat-card"
import { GlobalNetworkMap } from "@/components/global-network-map"
import { Activity, Zap, DollarSign } from "lucide-react"
import { fetchNodes, fetchStats } from "@/lib/api"
import type { NodeInfo, NetworkStats } from "@/lib/types"

const DEFAULT_STATS = [
  {
    icon: Activity,
    label: "Total Active Nodes",
    value: "0",
    sub: "nodes online",
    color: "hsl(142, 71%, 45%)",
  },
  {
    icon: Zap,
    label: "Current Network TPS",
    value: "0",
    sub: "tokens/sec",
    color: "hsl(48, 96%, 53%)",
  },
  {
    icon: DollarSign,
    label: "Avg Cost / 1k Tokens",
    value: "$0.0004",
    sub: "per 1k tokens",
    color: "hsl(217, 91%, 60%)",
  },
]

export function NetworkExplorer() {
  const [draftNodes, setDraftNodes] = useState<NodeInfo[]>([])
  const [targetNodes, setTargetNodes] = useState<NodeInfo[]>([])
  const [stats, setStats] = useState(DEFAULT_STATS)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const nodes = await fetchNodes()
        const netStats = await fetchStats()
        if (cancelled) return

        setDraftNodes(nodes.filter(n => n.type === "draft"))
        setTargetNodes(nodes.filter(n => n.type === "target"))

        setStats([
          {
            icon: Activity,
            label: "Total Active Nodes",
            value: String(netStats.active_draft_nodes + netStats.active_target_nodes),
            sub: "nodes online",
            color: "hsl(142, 71%, 45%)",
          },
          {
            icon: Zap,
            label: "Current Network TPS",
            value: netStats.total_tps > 0 ? netStats.total_tps.toLocaleString() : "0",
            sub: "tokens/sec",
            color: "hsl(48, 96%, 53%)",
          },
          {
            icon: DollarSign,
            label: "Avg Cost / 1k Tokens",
            value: `$${netStats.avg_cost_per_1k.toFixed(4)}`,
            sub: "per 1k tokens",
            color: "hsl(217, 91%, 60%)",
          },
        ])
        setLoaded(true)
      } catch {
        // Backend offline — keep defaults
      }
    }

    load()
    const interval = setInterval(load, 30000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return (
    <div className="flex flex-1 flex-col gap-6 overflow-y-auto p-6">
      {/* Page header */}
      <div>
        <h2 className="font-heading text-2xl font-bold tracking-tight text-foreground">Network</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Live view of available Draft nodes.
        </p>
      </div>

      {/* Stats row */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {stats.map((stat, i) => (
          <NetworkStatCard
            key={stat.label}
            {...stat}
            index={i}
            breakdown={i === 0 ? { draft: draftNodes.length, target: targetNodes.length } : undefined}
          />
        ))}
      </div>

      {/* Map with Target/Draft panels on the right */}
      <div className="flex gap-4">
        <div className="min-w-0 flex-1">
          <GlobalNetworkMap />
        </div>
        <div className="flex w-[26rem] shrink-0 flex-col gap-4 xl:w-[30rem] 2xl:w-[34rem]">
          <TargetNodesTable nodes={loaded ? targetNodes : undefined} />
          <DraftNodesTable nodes={loaded ? draftNodes : undefined} />
        </div>
      </div>
    </div>
  )
}
