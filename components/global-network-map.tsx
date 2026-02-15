"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Globe, Cpu, Zap, TrendingUp, Clock, Activity } from "lucide-react"
import dynamic from "next/dynamic"

// Dynamically import map components to avoid SSR issues
const MapContainer = dynamic(
  () => import("react-leaflet").then((mod) => mod.MapContainer),
  { ssr: false }
)
const TileLayer = dynamic(
  () => import("react-leaflet").then((mod) => mod.TileLayer),
  { ssr: false }
)
const CircleMarker = dynamic(
  () => import("react-leaflet").then((mod) => mod.CircleMarker),
  { ssr: false }
)
const Polyline = dynamic(
  () => import("react-leaflet").then((mod) => mod.Polyline),
  { ssr: false }
)
const Popup = dynamic(
  () => import("react-leaflet").then((mod) => mod.Popup),
  { ssr: false }
)

interface NodeLocation {
  lat: number
  lng: number
  city: string
  country: string
}

interface Node {
  id: string
  type: "draft" | "target"
  hardware: string
  model: string
  status: "online" | "offline" | "busy"
  latency: number
  price: number
  gpu_memory: string
  location: NodeLocation
  earnings: number
  uptime: number
}

interface Connection {
  from: string
  to: string
  type: "potential" | "active"
}

interface ActiveRoute {
  from: string
  to: string
  type: "active"
  tokens: number
  acceptance_rate: number
  timestamp: string
}

interface ConnectionData {
  static_connections: Connection[]
  active_route: ActiveRoute | null
  total_inferences: number
}

export function GlobalNetworkMap() {
  const [nodes, setNodes] = useState<Node[]>([])
  const [connections, setConnections] = useState<ConnectionData>({
    static_connections: [],
    active_route: null,
    total_inferences: 0,
  })
  const [isClient, setIsClient] = useState(false)
  const [showDemoNodes, setShowDemoNodes] = useState(false)

  useEffect(() => {
    setIsClient(true)

    // Fetch nodes and connections on mount
    const fetchData = () => {
      const demoParam = showDemoNodes ? "?include_demo=true" : ""

      fetch(`http://localhost:8000/api/nodes${demoParam}`)
        .then((res) => res.json())
        .then((data) => setNodes(data))
        .catch((err) => console.error("Failed to fetch nodes:", err))

      fetch(`http://localhost:8000/api/network/connections${demoParam}`)
        .then((res) => res.json())
        .then((data) => setConnections(data))
        .catch((err) => console.error("Failed to fetch connections:", err))
    }

    fetchData()

    // Poll for updates every 2 seconds (faster to catch active routes)
    const interval = setInterval(fetchData, 2000)

    return () => clearInterval(interval)
  }, [showDemoNodes])

  const getNodeById = (id: string) => nodes.find((n) => n.id === id)

  const getNodeColor = (node: Node) => {
    if (node.status === "offline") return "#64748b" // gray
    if (node.status === "busy") return "#f59e0b" // amber
    return node.type === "target" ? "#3b82f6" : "#10b981" // blue for target, green for draft
  }

  const getNodeSize = (node: Node) => {
    return node.type === "target" ? 16 : 12
  }

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "online":
        return "border-primary/30 bg-primary/10 text-primary"
      case "busy":
        return "border-yellow-500/30 bg-yellow-500/10 text-yellow-400"
      case "offline":
        return "border-gray-500/30 bg-gray-500/10 text-gray-400"
      default:
        return ""
    }
  }

  const onlineNodes = nodes.filter((n) => n.status === "online" || n.status === "busy")
  const totalNodes = nodes.length
  const draftNodes = nodes.filter((n) => n.type === "draft").length
  const targetNodes = nodes.filter((n) => n.type === "target").length

  if (!isClient) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.4 }}
      >
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 font-heading text-sm font-medium text-muted-foreground">
              <Globe className="h-4 w-4 text-blue-400" />
              Global Speculative GPU Network
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex h-96 items-center justify-center text-sm text-muted-foreground">
              Loading map...
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.4 }}
    >
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 font-heading text-sm font-medium text-muted-foreground">
              <Globe className="h-4 w-4 text-blue-400" />
              Global Speculative GPU Network
            </CardTitle>
            <div className="flex items-center gap-4 text-xs">
              {/* Demo Toggle */}
              <button
                onClick={() => setShowDemoNodes(!showDemoNodes)}
                className={`flex items-center gap-2 rounded-md border px-3 py-1.5 transition-colors ${
                  showDemoNodes
                    ? "border-primary/30 bg-primary/10 text-primary"
                    : "border-border/40 bg-secondary/20 text-muted-foreground hover:bg-secondary/30"
                }`}
              >
                <div className={`h-2 w-2 rounded-full ${showDemoNodes ? "bg-primary animate-pulse" : "bg-muted-foreground"}`}></div>
                <span className="text-[11px] font-medium">
                  {showDemoNodes ? "Demo Mode ON" : "Real Nodes Only"}
                </span>
              </button>

              <div className="h-4 w-px bg-border/40"></div>

              <div className="flex items-center gap-1.5">
                <div className="h-2.5 w-2.5 rounded-full bg-green-500"></div>
                <span className="text-muted-foreground">
                  {draftNodes} Draft
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-2.5 w-2.5 rounded-full bg-blue-500"></div>
                <span className="text-muted-foreground">
                  {targetNodes} Target
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="font-semibold text-foreground">{onlineNodes.length}</span>
                <span className="text-muted-foreground">/ {totalNodes} online</span>
              </div>
              {connections.active_route && (
                <div className="flex items-center gap-1.5 rounded-full border border-green-500/30 bg-green-500/10 px-2 py-1">
                  <Activity className="h-3 w-3 animate-pulse text-green-400" />
                  <span className="text-green-400 font-semibold">LIVE</span>
                </div>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-hidden rounded-lg border border-border/40">
            <MapContainer
              center={[20, 0]}
              zoom={2}
              style={{ height: "500px", width: "100%" }}
              className="z-0"
            >
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
              />

              {/* Draw connection lines */}
              {connections.static_connections.map((conn, idx) => {
                const fromNode = getNodeById(conn.from)
                const toNode = getNodeById(conn.to)

                if (!fromNode || !toNode) return null

                const isActive =
                  connections.active_route &&
                  connections.active_route.from === conn.from &&
                  connections.active_route.to === conn.to

                return (
                  <Polyline
                    key={`${conn.from}-${conn.to}-${idx}`}
                    positions={[
                      [fromNode.location.lat, fromNode.location.lng],
                      [toNode.location.lat, toNode.location.lng],
                    ]}
                    pathOptions={{
                      color: isActive ? "#10b981" : "#3b82f6",
                      weight: isActive ? 3 : 1,
                      opacity: isActive ? 0.8 : 0.15,
                      dashArray: isActive ? undefined : "5, 10",
                    }}
                  >
                    {isActive && connections.active_route && (
                      <Popup>
                        <div className="space-y-2 p-1">
                          <div className="flex items-center gap-2">
                            <Activity className="h-4 w-4 text-green-400" />
                            <span className="font-heading text-sm font-semibold text-green-400">
                              Active Inference
                            </span>
                          </div>
                          <div className="space-y-1 text-xs">
                            <div>
                              <span className="text-muted-foreground">Route: </span>
                              <span className="text-foreground font-mono">
                                {fromNode.location.city} → {toNode.location.city}
                              </span>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Tokens: </span>
                              <span className="text-foreground font-mono">
                                {connections.active_route.tokens}
                              </span>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Acceptance: </span>
                              <span className="text-foreground font-mono">
                                {(connections.active_route.acceptance_rate * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </Popup>
                    )}
                  </Polyline>
                )
              })}

              {/* Draw nodes on top of connections */}
              {nodes.map((node) => (
                <CircleMarker
                  key={node.id}
                  center={[node.location.lat, node.location.lng]}
                  radius={getNodeSize(node)}
                  fillColor={getNodeColor(node)}
                  color="#fff"
                  weight={2}
                  opacity={node.status === "offline" ? 0.4 : 0.8}
                  fillOpacity={node.status === "offline" ? 0.3 : 0.7}
                >
                  <Popup className="custom-popup">
                    <div className="min-w-64 space-y-3 p-2">
                      {/* Header */}
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <h3 className="font-heading text-sm font-semibold text-foreground">
                            {node.location.city}
                          </h3>
                          <p className="text-xs text-muted-foreground">
                            {node.location.country}
                          </p>
                        </div>
                        <Badge
                          variant="outline"
                          className={`text-[10px] ${getStatusBadgeClass(node.status)}`}
                        >
                          {node.status}
                        </Badge>
                      </div>

                      {/* Node Type Badge */}
                      <div className="flex items-center gap-2">
                        <Badge
                          variant="outline"
                          className={`text-[10px] ${
                            node.type === "target"
                              ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
                              : "border-green-500/30 bg-green-500/10 text-green-400"
                          }`}
                        >
                          {node.type === "target" ? "🎯 Target Node" : "⚡ Draft Node"}
                        </Badge>
                      </div>

                      {/* Hardware Info */}
                      <div className="space-y-1.5 rounded-md border border-border/40 bg-secondary/30 p-2">
                        <div className="flex items-center gap-1.5 text-xs">
                          <Cpu className="h-3 w-3 text-muted-foreground" />
                          <span className="font-medium text-foreground">{node.hardware}</span>
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                          {node.gpu_memory} • {node.model.split("/")[1]}
                        </div>
                      </div>

                      {/* Stats Grid */}
                      <div className="grid grid-cols-2 gap-2">
                        <div className="rounded-md border border-border/40 bg-secondary/20 p-2">
                          <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                            <Zap className="h-3 w-3" />
                            Latency
                          </div>
                          <div className="mt-0.5 font-mono text-sm font-bold text-foreground">
                            {node.latency}ms
                          </div>
                        </div>
                        <div className="rounded-md border border-border/40 bg-secondary/20 p-2">
                          <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                            <Clock className="h-3 w-3" />
                            Uptime
                          </div>
                          <div className="mt-0.5 font-mono text-sm font-bold text-foreground">
                            {node.uptime.toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      {/* Earnings */}
                      {node.earnings > 0 && (
                        <div className="rounded-md border border-border/40 bg-primary/5 p-2">
                          <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                            <TrendingUp className="h-3 w-3" />
                            Total Earnings
                          </div>
                          <div className="mt-0.5 font-mono text-sm font-bold text-primary">
                            ${node.earnings.toFixed(2)}
                          </div>
                        </div>
                      )}

                      {/* Price */}
                      <div className="border-t border-border/40 pt-2 text-center text-[10px] text-muted-foreground">
                        ${node.price.toFixed(3)}/1K tokens
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>

          {/* Legend and Stats */}
          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center gap-6 text-xs text-muted-foreground">
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-green-500"></div>
                <span>Draft (Online)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-blue-500"></div>
                <span>Target (Online)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-amber-500"></div>
                <span>Busy</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-gray-500"></div>
                <span>Offline</span>
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              <span className="font-semibold text-foreground">
                {connections.total_inferences}
              </span>{" "}
              total inferences processed
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
