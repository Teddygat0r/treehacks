"use client"

import { useState, useEffect } from "react"
import { Key, Copy, Check, Eye, EyeOff, RefreshCw } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

const API_KEY_STORAGE_KEY = "nexus_api_key"

export function ApiCredentialsCard() {
  const [apiKey, setApiKey] = useState<string | null>(null)
  const [showKey, setShowKey] = useState(false)
  const [copied, setCopied] = useState(false)

  // Load API key from localStorage on mount
  useEffect(() => {
    const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY)
    if (storedKey) {
      setApiKey(storedKey)
    }
  }, [])

  const handleCopy = async () => {
    if (!apiKey) return
    await navigator.clipboard.writeText(apiKey)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleGenerate = () => {
    // Generate a new demo key
    const randomKey = `nx_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`
    setApiKey(randomKey)
    setShowKey(true)
    // Persist to localStorage
    localStorage.setItem(API_KEY_STORAGE_KEY, randomKey)
    // Dispatch custom event to notify other components
    window.dispatchEvent(new Event("nexus-api-key-updated"))
  }

  const displayKey = apiKey ? (showKey ? apiKey : "nx_" + "•".repeat(28)) : null

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardContent className="p-5">
        <div className="mb-4 flex items-center gap-3">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-lg"
            style={{
              backgroundColor: "#3b82f615",
              boxShadow: "inset 0 0 0 1px #3b82f630",
            }}
          >
            <Key size={20} color="#3b82f6" strokeWidth={2} />
          </div>
          <div>
            <h3 className="font-heading text-sm font-semibold text-foreground">
              API Credentials
            </h3>
            <p className="text-xs text-muted-foreground">
              Use this key to access the network
            </p>
          </div>
        </div>

        <div className="mb-3 space-y-2">
          <label className="text-xs font-medium text-muted-foreground">
            API Key
          </label>
          {displayKey ? (
            <div className="flex items-center gap-2">
              <div className="flex flex-1 items-center gap-2 rounded-md border border-border/50 bg-background/50 px-3 py-2">
                <code className="flex-1 font-mono text-xs text-foreground">
                  {displayKey}
                </code>
                <button
                  onClick={() => setShowKey(!showKey)}
                  className="text-muted-foreground transition-colors hover:text-foreground"
                  title={showKey ? "Hide key" : "Show key"}
                >
                  {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              <button
                onClick={handleCopy}
                className="rounded-md border border-border/50 bg-background/50 p-2 text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
                title="Copy to clipboard"
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
          ) : (
            <div className="flex items-center justify-center rounded-md border border-border/50 bg-background/30 px-3 py-8">
              <span className="text-xs text-muted-foreground">
                No API key generated yet
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleGenerate}
            className="flex items-center gap-2 rounded-md bg-blue-500 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-blue-600"
          >
            <RefreshCw size={12} />
            {apiKey ? "Generate New Key" : "Generate API Key"}
          </button>
          {apiKey && (
            <span className="text-xs text-muted-foreground">
              Keep your key secure
            </span>
          )}
        </div>

        <div className="mt-4 rounded-md border border-border/50 bg-background/30 p-3">
          <div className="text-xs text-muted-foreground">
            <p className="mb-1 font-medium">Base URL:</p>
            <code className="font-mono text-foreground">
              http://localhost:8000
            </code>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
