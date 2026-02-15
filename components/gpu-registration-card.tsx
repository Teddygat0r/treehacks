"use client"

import { useState } from "react"
import { Server, Plus, AlertCircle } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

type NodeType = "draft" | "target"

export function GpuRegistrationCard() {
  const [nodeType, setNodeType] = useState<NodeType>("draft")
  const [isRegistering, setIsRegistering] = useState(false)

  const handleRegister = async () => {
    setIsRegistering(true)
    // Simulate registration
    await new Promise((resolve) => setTimeout(resolve, 1500))
    setIsRegistering(false)
    alert(`GPU registered as ${nodeType} node! (Demo)`)
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardContent className="p-5">
        <div className="mb-4 flex items-center gap-3">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-lg"
            style={{
              backgroundColor: "#22c55e15",
              boxShadow: "inset 0 0 0 1px #22c55e30",
            }}
          >
            <Server size={20} color="#22c55e" strokeWidth={2} />
          </div>
          <div>
            <h3 className="font-heading text-sm font-semibold text-foreground">
              Register Your GPU
            </h3>
            <p className="text-xs text-muted-foreground">
              Add your GPU to earn rewards
            </p>
          </div>
        </div>

        <div className="mb-4 space-y-3">
          <div>
            <label className="mb-2 block text-xs font-medium text-muted-foreground">
              Node Type
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setNodeType("draft")}
                className={`rounded-md border p-3 text-left transition-colors ${
                  nodeType === "draft"
                    ? "border-green-500/50 bg-green-500/10"
                    : "border-border/50 bg-background/30 hover:border-border"
                }`}
              >
                <div className="mb-1 text-xs font-semibold text-foreground">
                  Draft Node
                </div>
                <div className="text-[10px] text-muted-foreground">
                  Run small models (1-3B), low memory
                </div>
              </button>
              <button
                onClick={() => setNodeType("target")}
                className={`rounded-md border p-3 text-left transition-colors ${
                  nodeType === "target"
                    ? "border-blue-500/50 bg-blue-500/10"
                    : "border-border/50 bg-background/30 hover:border-border"
                }`}
              >
                <div className="mb-1 text-xs font-semibold text-foreground">
                  Target Node
                </div>
                <div className="text-[10px] text-muted-foreground">
                  Run large models (7B+), high memory
                </div>
              </button>
            </div>
          </div>

          <div className="rounded-md border border-border/50 bg-background/30 p-3 text-xs">
            <div className="mb-2 flex items-start gap-2">
              <AlertCircle size={14} className="mt-0.5 text-blue-500" />
              <div>
                <div className="font-medium text-foreground">Requirements</div>
                <ul className="mt-1 space-y-0.5 text-muted-foreground">
                  {nodeType === "draft" ? (
                    <>
                      <li>• NVIDIA GPU with 8GB+ VRAM</li>
                      <li>• CUDA 11.8+ installed</li>
                      <li>• ~2GB disk space for model</li>
                    </>
                  ) : (
                    <>
                      <li>• NVIDIA GPU with 24GB+ VRAM</li>
                      <li>• CUDA 11.8+ installed</li>
                      <li>• ~10GB disk space for model</li>
                    </>
                  )}
                </ul>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleRegister}
          disabled={isRegistering}
          className="flex w-full items-center justify-center gap-2 rounded-md bg-green-500 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-green-600 disabled:opacity-50"
        >
          {isRegistering ? (
            <>
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              Registering...
            </>
          ) : (
            <>
              <Plus size={16} />
              Register GPU
            </>
          )}
        </button>

        <p className="mt-3 text-center text-xs text-muted-foreground">
          Estimated earnings: ${nodeType === "draft" ? "2-5" : "8-15"}/day
        </p>
      </CardContent>
    </Card>
  )
}
