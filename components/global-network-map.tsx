"use client"

import { Fragment, useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Globe, Cpu, Zap, TrendingUp, Clock, Activity } from "lucide-react"
import dynamic from "next/dynamic"
import { useMap } from "react-leaflet"
import L from "leaflet"

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

interface AnimatedTransmission {
  id: string
  from: string
  to: string
  startTime: number
}

/** Returns points along a slight curve between a and b (quadratic Bézier). reverse = true bulges the opposite side with a stronger curve so it doesn't overlap (for return/verification path). */
function curvedLine(
  a: [number, number],
  b: [number, number],
  numPoints = 14,
  reverse = false
): [number, number][] {
  const [lat0, lng0] = a
  const [lat1, lng1] = b
  const midLat = (lat0 + lat1) / 2
  const midLng = (lng0 + lng1) / 2
  const perpLat = lng0 - lng1
  const perpLng = lat1 - lat0
  const len = Math.sqrt(perpLat * perpLat + perpLng * perpLng) || 1
  const dist = Math.sqrt((lat1 - lat0) ** 2 + (lng1 - lng0) ** 2)
  const baseStrength = reverse ? 0.14 : 0.10
  const curveStrength = dist * baseStrength * (reverse ? -1 : 1)
  const ctrlLat = midLat + (perpLat / len) * curveStrength
  const ctrlLng = midLng + (perpLng / len) * curveStrength
  const points: [number, number][] = []
  for (let i = 0; i <= numPoints; i++) {
    const t = i / numPoints
    const u = 1 - t
    const lat = u * u * lat0 + 2 * u * t * ctrlLat + t * t * lat1
    const lng = u * u * lng0 + 2 * u * t * ctrlLng + t * t * lng1
    points.push([lat, lng])
  }
  return points
}

function FitBoundsToNodes({
  positions,
}: {
  positions: [number, number][]
}) {
  const map = useMap()
  useEffect(() => {
    if (positions.length === 0) return
    if (positions.length === 1) {
      map.setView(positions[0], 10)
      return
    }
    const bounds = L.latLngBounds(positions)
    map.fitBounds(bounds, { padding: [24, 24], maxZoom: 12 })
  }, [map, positions])
  return null
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
  const [animatedTransmissions, setAnimatedTransmissions] = useState<AnimatedTransmission[]>([])

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

  // Simulate active data transmissions with 7-second duration
  useEffect(() => {
    if (!showDemoNodes || connections.static_connections.length === 0) return

    const addNewTransmission = () => {
      const allConnections = connections.static_connections
      if (allConnections.length === 0) return
      const conn = allConnections[Math.floor(Math.random() * allConnections.length)]
      setAnimatedTransmissions((prev) => [
        ...prev,
        {
          id: `${conn.from}-${conn.to}-${Date.now()}-${Math.random()}`,
          from: conn.from,
          to: conn.to,
          startTime: Date.now(),
        },
      ])
    }

    addNewTransmission()
    const interval = setInterval(addNewTransmission, 2800) // One new transmission every ~2.8s for a continuous trickle

    return () => clearInterval(interval)
  }, [showDemoNodes, connections.static_connections])

  // Clean up old transmissions after 7 seconds
  useEffect(() => {
    const cleanupInterval = setInterval(() => {
      const now = Date.now()
      setAnimatedTransmissions((prev) =>
        prev.filter((t) => now - t.startTime < 7000)
      )
    }, 500) // Check every 500ms

    return () => clearInterval(cleanupInterval)
  }, [])

  const getNodeById = (id: string) => nodes.find((n) => n.id === id)

  const getNodeColor = (node: Node) => {
    // Just differentiate between target and draft nodes
    return node.type === "target" ? "#3b82f6" : "#10b981" // blue for target, green for draft
  }

  const getNodeSize = (node: Node) => {
    return node.type === "target" ? 12 : 6
  }

  // Get adjusted position for nodes to prevent overlap at same location
  const getNodePosition = (node: Node): [number, number] => {
    const { lat, lng } = node.location

    // Find all nodes at the same location
    const nodesAtLocation = nodes.filter(
      (n) => n.location.lat === lat && n.location.lng === lng
    )

    // If only one node at this location, return as-is
    if (nodesAtLocation.length === 1) {
      return [lat, lng]
    }

    // Multiple nodes at same location - apply offset
    const index = nodesAtLocation.findIndex((n) => n.id === node.id)
    const offsetDistance = 0.08 // degrees

    // Arrange in a circle around the original point
    const angle = (index * 2 * Math.PI) / nodesAtLocation.length
    const offsetLat = lat + offsetDistance * Math.cos(angle)
    const offsetLng = lng + offsetDistance * Math.sin(angle)

    return [offsetLat, offsetLng]
  }

  const nodePositions = useMemo(
    () => nodes.map((node) => getNodePosition(node)),
    [nodes]
  )

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

  // All nodes are shown as online in the visualization
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
              Global Distributed GPU Network
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
    <>
      <style jsx global>{`
        @keyframes dash {
          to {
            stroke-dashoffset: -20;
          }
        }
        .animate-dash {
          animation: dash 0.8s linear infinite;
        }
        /* Transmission: fade in (0–0.4s), hold, fade out (last 0.5s) over 7s */
        @keyframes transmission-lifecycle {
          0% {
            opacity: 0;
            stroke-width: 0.8;
          }
          5.7% {
            opacity: 0.85;
            stroke-width: 1.8;
          }
          92.8% {
            opacity: 0.85;
            stroke-width: 1.8;
          }
          100% {
            opacity: 0;
            stroke-width: 0.8;
          }
        }
        .transmission-line {
          animation:
            dash 0.8s linear infinite,
            transmission-lifecycle 7s ease-in-out forwards;
          transform-origin: center;
        }
        /* Real inference route: gentle fade-in when it appears */
        @keyframes real-inference-fade-in {
          0% {
            opacity: 0;
          }
          100% {
            opacity: 1;
          }
        }
        .real-inference-appear {
          animation: real-inference-fade-in 0.6s ease-out forwards;
        }
      `}</style>
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
              Global Distributed GPU Network
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
                  {draftNodes} Draft Nodes
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-2.5 w-2.5 rounded-full bg-blue-500"></div>
                <span className="text-muted-foreground">
                  {targetNodes} Target Nodes
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="font-semibold text-foreground">{totalNodes}</span>
                <span className="text-muted-foreground">total nodes</span>
              </div>
              {connections.active_route && (
                <div className="flex items-center gap-1.5 rounded-full border border-yellow-500/30 bg-yellow-500/10 px-2 py-1">
                  <Activity className="h-3 w-3 animate-pulse text-yellow-400" />
                  <span className="text-yellow-400 font-semibold">LIVE</span>
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

              <FitBoundsToNodes positions={nodePositions} />

              {/* Draw static connection lines (visible network topology) */}
              {connections.static_connections.map((conn, idx) => {
                const fromNode = getNodeById(conn.from)
                const toNode = getNodeById(conn.to)

                if (!fromNode || !toNode) return null

                // Check if this is the real active inference route
                const isRealActive =
                  connections.active_route &&
                  connections.active_route.from === conn.from &&
                  connections.active_route.to === conn.to

                return (
                  <Fragment
                    key={
                      isRealActive && connections.active_route
                        ? `active-${connections.active_route.from}-${connections.active_route.to}-${connections.active_route.timestamp}`
                        : `static-${conn.from}-${conn.to}-${idx}`
                    }
                  >
                    <Polyline
                      key={`${conn.from}-${conn.to}-${idx}-fwd`}
                      positions={curvedLine(
                        getNodePosition(fromNode),
                        getNodePosition(toNode)
                      )}
                      pathOptions={{
                        color: isRealActive ? "#fbbf24" : "#3b82f6",
                        weight: isRealActive ? 1.8 : 1,
                        opacity: isRealActive ? 0.85 : 0.15,
                        dashArray: isRealActive ? "8, 12" : "3, 8",
                        ...(isRealActive && {
                          className: "animate-dash real-inference-appear",
                        }),
                      }}
                    >
                      {isRealActive && connections.active_route && (
                        <Popup>
                          <div className="space-y-2 p-1">
                            <div className="flex items-center gap-2">
                              <Activity className="h-4 w-4 text-yellow-400" />
                              <span className="font-heading text-sm font-semibold text-yellow-400">
                                Real Inference
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
                    {/* Verification path (backwards) for real inference */}
                    {isRealActive && (
                      <Polyline
                        key={`active-verify-${conn.from}-${conn.to}-${connections.active_route?.timestamp ?? ""}`}
                        positions={curvedLine(
                          getNodePosition(toNode),
                          getNodePosition(fromNode),
                          14,
                          true
                        )}
                        pathOptions={{
                          color: "#22d3ee",
                          weight: 1.4,
                          opacity: 0.8,
                          dashArray: "6, 10",
                          className: "animate-dash real-inference-appear",
                        }}
                      />
                    )}
                  </Fragment>
                )
              })}

              {/* Draw animated transmission lines (yellow, flowing) */}
              {animatedTransmissions.map((transmission) => {
                const fromNode = getNodeById(transmission.from)
                const toNode = getNodeById(transmission.to)

                if (!fromNode || !toNode) return null

                const fromPos = getNodePosition(fromNode)
                const toPos = getNodePosition(toNode)

                return (
                  <Fragment key={transmission.id}>
                    <Polyline
                      key={`${transmission.id}-fwd`}
                      positions={curvedLine(fromPos, toPos)}
                      pathOptions={{
                        color: "#fbbf24",
                        weight: 1.8,
                        opacity: 1,
                        dashArray: "8, 12",
                        className: "animate-dash transmission-line",
                      }}
                    />
                    {/* Verification path (backwards, mirrored + different color) */}
                    <Polyline
                      key={`${transmission.id}-verify`}
                      positions={curvedLine(toPos, fromPos, 14, true)}
                      pathOptions={{
                        color: "#22d3ee",
                        weight: 1.4,
                        opacity: 1,
                        dashArray: "6, 10",
                        className: "animate-dash transmission-line",
                      }}
                    />
                  </Fragment>
                )
              })}

              {/* Draw nodes on top of connections */}
              {nodes.map((node) => (
                <CircleMarker
                  key={node.id}
                  center={getNodePosition(node)}
                  radius={getNodeSize(node)}
                  fillColor={getNodeColor(node)}
                  color="#fff"
                  weight={0.8}
                  opacity={0.6}
                  fillOpacity={0.9}
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
                <span>Draft Nodes</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-blue-500"></div>
                <span>Target Nodes</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-yellow-500"></div>
                <span>Inference / Data</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-cyan-400"></div>
                <span>Verification</span>
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              <span className="font-semibold text-foreground">
                {connections.total_inferences}
              </span>{" "}
              real inferences • <span className="font-semibold text-yellow-500">
                {animatedTransmissions.length}
              </span>{" "}
              active transmissions
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
    </>
  )
}
