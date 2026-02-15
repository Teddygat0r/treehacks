"use client"

import { useState } from "react"
import { Activity, Settings, Circle } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface NodeConfig {
  model: string
  pricePerToken: number
  status: "online" | "offline"
}

export function NodeStatusCard() {
  const [hasRegisteredGpu, setHasRegisteredGpu] = useState(false)
  const [nodeConfig, setNodeConfig] = useState<NodeConfig>({
    model: "Qwen/Qwen2.5-1.5B-Instruct",
    pricePerToken: 0.00005,
    status: "online",
  })

  // Mock: Show registration state
  if (!hasRegisteredGpu) {
    return (
      <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
        <CardContent className="p-5">
          <div className="flex min-h-[200px] flex-col items-center justify-center text-center">
            <div className="mb-3 rounded-full bg-muted/50 p-4">
              <Activity size={32} className="text-muted-foreground" />
            </div>
            <h3 className="mb-2 font-heading text-sm font-semibold text-foreground">
              No Active Nodes
            </h3>
            <p className="mb-4 text-xs text-muted-foreground">
              Register your GPU to start earning
            </p>
            <button
              onClick={() => setHasRegisteredGpu(true)}
              className="rounded-md bg-green-500 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-green-600"
            >
              Register GPU (Demo)
            </button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardContent className="p-5">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="flex h-10 w-10 items-center justify-center rounded-lg"
              style={{
                backgroundColor: "#3b82f615",
                boxShadow: "inset 0 0 0 1px #3b82f630",
              }}
            >
              <Activity size={20} color="#3b82f6" strokeWidth={2} />
            </div>
            <div>
              <h3 className="font-heading text-sm font-semibold text-foreground">
                Node Status
              </h3>
              <p className="text-xs text-muted-foreground">
                Monitor and configure
              </p>
            </div>
          </div>
          <Badge
            variant="outline"
            className="border-green-500/30 bg-green-500/10 text-green-500"
          >
            <Circle size={8} className="mr-1 fill-green-500" />
            Online
          </Badge>
        </div>

        <div className="space-y-3">
          {/* Hardware Info */}
          <div className="rounded-lg border border-border/50 bg-background/30 p-3">
            <div className="mb-2 text-xs font-medium text-foreground">
              Hardware
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">GPU:</span>{" "}
                <span className="font-medium text-foreground">RTX 4090</span>
              </div>
              <div>
                <span className="text-muted-foreground">VRAM:</span>{" "}
                <span className="font-medium text-foreground">24 GB</span>
              </div>
              <div>
                <span className="text-muted-foreground">Uptime:</span>{" "}
                <span className="font-medium text-foreground">99.8%</span>
              </div>
              <div>
                <span className="text-muted-foreground">Requests:</span>{" "}
                <span className="font-medium text-foreground">1,247</span>
              </div>
            </div>
          </div>

          {/* Configuration */}
          <div className="rounded-lg border border-border/50 bg-background/30 p-3">
            <div className="mb-3 flex items-center gap-2">
              <Settings size={14} className="text-muted-foreground" />
              <span className="text-xs font-medium text-foreground">
                Configuration
              </span>
            </div>

            <div className="space-y-2">
              <div>
                <label className="mb-1 block text-xs text-muted-foreground">
                  Model
                </label>
                <select
                  value={nodeConfig.model}
                  onChange={(e) =>
                    setNodeConfig({ ...nodeConfig, model: e.target.value })
                  }
                  className="w-full rounded-md border border-border/50 bg-background px-2 py-1.5 text-xs text-foreground"
                >
                  <option value="Qwen/Qwen2.5-1.5B-Instruct">
                    Qwen 2.5 1.5B
                  </option>
                  <option value="Qwen/Qwen2.5-3B-Instruct">
                    Qwen 2.5 3B
                  </option>
                  <option value="meta-llama/Llama-2-7b">Llama 2 7B</option>
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs text-muted-foreground">
                  Price per Token
                </label>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">$</span>
                  <input
                    type="number"
                    step="0.000001"
                    value={nodeConfig.pricePerToken}
                    onChange={(e) =>
                      setNodeConfig({
                        ...nodeConfig,
                        pricePerToken: parseFloat(e.target.value),
                      })
                    }
                    className="flex-1 rounded-md border border-border/50 bg-background px-2 py-1.5 text-xs text-foreground"
                  />
                </div>
              </div>
            </div>
          </div>

          <button className="w-full rounded-md border border-border/50 bg-background px-3 py-2 text-xs font-medium text-foreground transition-colors hover:bg-background/80">
            Save Configuration
          </button>
        </div>
      </CardContent>
    </Card>
  )
}
