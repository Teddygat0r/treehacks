"use client"

import { useState, useEffect } from "react"
import { DraftNodesTable } from "@/components/draft-nodes-table"
import { fetchNodes } from "@/lib/api"
import type { NodeInfo } from "@/lib/types"

export function NetworkExplorer() {
  const [draftNodes, setDraftNodes] = useState<NodeInfo[]>([])
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const nodes = await fetchNodes()
        if (cancelled) return

        setDraftNodes(nodes.filter(n => n.type === "draft"))
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

      {/* Nodes table */}
      <DraftNodesTable nodes={loaded ? draftNodes : undefined} />
    </div>
  )
}
