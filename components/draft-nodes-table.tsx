"use client"

import { useState, useMemo } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Cpu, ChevronDown } from "lucide-react"
import type { NodeInfo } from "@/lib/types"

const DEFAULT_DRAFT_NODES = [
  { id: "node_j9u1", hardware: "RTX 4070", latency: 8, price: 0.35 },
  { id: "node_p1d3", hardware: "RTX 4060", latency: 11, price: 0.25 },
  { id: "node_t6y4", hardware: "Mac M3", latency: 12, price: 0.20 },
  { id: "node_w4n6", hardware: "RTX 4090", latency: 6, price: 0.75 },
  { id: "node_a2f1", hardware: "RTX 3070", latency: 14, price: 0.22 },
  { id: "node_m5e9", hardware: "RTX 3080", latency: 13, price: 0.30 },
  { id: "node_7x9b", hardware: "RTX 3060", latency: 18, price: 0.15 },
  { id: "node_q3w7", hardware: "RTX 3060", latency: 21, price: 0.14 },
  { id: "node_b8g3", hardware: "RTX 3070 Ti", latency: 13, price: 0.24 },
  { id: "node_k4c8", hardware: "Mac M2", latency: 22, price: 0.12 },
  { id: "node_r8v2", hardware: "Mac M1 Pro", latency: 26, price: 0.10 },
  { id: "node_h2b5", hardware: "RX 7600", latency: 23, price: 0.13 },
  { id: "node_v1x8", hardware: "RTX 4060 Ti", latency: 9, price: 0.30 },
  { id: "node_c5z2", hardware: "Mac M3 Pro", latency: 10, price: 0.28 },
  { id: "node_f7s4", hardware: "RX 7700 XT", latency: 17, price: 0.18 },
  { id: "node_n3k9", hardware: "RTX 3090", latency: 11, price: 0.40 },
  { id: "node_d6p1", hardware: "Mac M4", latency: 7, price: 0.35 },
  { id: "node_g2r5", hardware: "Intel Arc A770", latency: 28, price: 0.08 },
  { id: "node_l8w3", hardware: "RTX 4080", latency: 7, price: 0.55 },
  { id: "node_s9m7", hardware: "RX 7800 XT", latency: 15, price: 0.20 },
]

const HARDWARE_TYPES = ["All Hardware", "RTX 3060", "RTX 3070", "RTX 3070 Ti", "RTX 3080", "RTX 3090", "RTX 4060", "RTX 4060 Ti", "RTX 4070", "RTX 4080", "RTX 4090", "Mac M1 Pro", "Mac M2", "Mac M3", "Mac M3 Pro", "Mac M4", "RX 7600", "RX 7700 XT", "RX 7800 XT", "Intel Arc A770"]
const LATENCY_OPTIONS = ["Any Latency", "< 10ms", "< 15ms", "< 25ms"]

interface DraftNodesTableProps {
  nodes?: NodeInfo[]
}

function PulsingDot() {
  return (
    <span className="relative flex h-2 w-2">
      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
      <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
    </span>
  )
}

export function DraftNodesTable({ nodes: liveNodes }: DraftNodesTableProps) {
  const [hardwareFilter, setHardwareFilter] = useState("All Hardware")
  const [latencyFilter, setLatencyFilter] = useState("Any Latency")
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  // Map live NodeInfo[] to the table format, or use defaults
  const DRAFT_NODES = useMemo(() => {
    if (!liveNodes) return DEFAULT_DRAFT_NODES
    return liveNodes.map(n => ({
      id: n.id,
      hardware: n.hardware,
      latency: n.latency,
      price: n.price,
    }))
  }, [liveNodes])

  const filtered = useMemo(() => {
    let nodes = DRAFT_NODES
    if (hardwareFilter !== "All Hardware") {
      nodes = nodes.filter((n) => n.hardware === hardwareFilter)
    }
    if (latencyFilter !== "Any Latency") {
      const max = parseInt(latencyFilter.replace(/[^0-9]/g, ""), 10)
      nodes = nodes.filter((n) => n.latency < max)
    }
    return nodes.sort((a, b) => a.latency - b.latency)
  }, [hardwareFilter, latencyFilter, DRAFT_NODES])

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.5 }}>
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 font-heading text-sm font-medium text-muted-foreground">
              <Cpu className="h-4 w-4 text-primary" />
              Draft Nodes (Edge)
            </CardTitle>
            <Badge variant="outline" className="border-primary/30 bg-primary/10 text-primary font-mono text-[10px]">
              {filtered.length} nodes
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="px-4 pb-4">
          {/* Filter bar */}
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <div className="relative">
              <select
                value={hardwareFilter}
                onChange={(e) => setHardwareFilter(e.target.value)}
                className="appearance-none rounded-md border border-border/60 bg-secondary/50 py-1.5 pl-3 pr-8 text-xs text-foreground outline-none transition-colors hover:bg-secondary focus:ring-1 focus:ring-ring"
              >
                {HARDWARE_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
            </div>
            <div className="relative">
              <select
                value={latencyFilter}
                onChange={(e) => setLatencyFilter(e.target.value)}
                className="appearance-none rounded-md border border-border/60 bg-secondary/50 py-1.5 pl-3 pr-8 text-xs text-foreground outline-none transition-colors hover:bg-secondary focus:ring-1 focus:ring-ring"
              >
                {LATENCY_OPTIONS.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
            </div>
          </div>

          {/* Table */}
          <div className="overflow-hidden rounded-lg border border-border/40">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/40 bg-secondary/30">
                  <th className="px-3 py-2 text-left font-medium text-muted-foreground">Node ID</th>
                  <th className="px-3 py-2 text-left font-medium text-muted-foreground">Hardware</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Latency</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground">Price/Hr</th>
                  <th className="px-3 py-2 text-right font-medium text-muted-foreground"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((node) => (
                  <tr
                    key={node.id}
                    className="border-b border-border/20 transition-colors hover:bg-secondary/20"
                  >
                    <td className="px-3 py-2.5">
                      <span className="flex items-center gap-2">
                        <PulsingDot />
                        <span className="font-mono text-foreground">{node.id}</span>
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-foreground">{node.hardware}</td>
                    <td className="px-3 py-2.5 text-right">
                      <span
                        className={`font-mono ${
                          node.latency < 10
                            ? "text-primary"
                            : node.latency < 20
                              ? "text-yellow-400"
                              : "text-orange-400"
                        }`}
                      >
                        {node.latency}ms
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono text-foreground">
                      ${node.price.toFixed(2)}/hr
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      <button
                        onClick={() => setSelectedNode(selectedNode === node.id ? null : node.id)}
                        className={`rounded-md px-3 py-1 text-[10px] font-semibold transition-colors ${
                          selectedNode === node.id
                            ? "border border-green-500/40 bg-green-500/15 text-green-400"
                            : "bg-primary text-primary-foreground hover:bg-primary/90"
                        }`}
                      >
                        {selectedNode === node.id ? "Selected" : "Rent"}
                      </button>
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-3 py-6 text-center text-muted-foreground">
                      No nodes match the selected filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
