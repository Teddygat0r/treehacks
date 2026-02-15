/**
 * Global inference service that maintains WebSocket connection
 * even when navigating between tabs
 */

import type { VisibleToken } from "@/components/chat-panel"
import type { StreamMessage, TokenEvent } from "@/lib/types"
import { InferenceStore } from "./inference-store"

type TokenCallback = (token: TokenEvent) => void
type RoundCallback = (acceptanceRate: number) => void
type DoneCallback = (data: any) => void
type ErrorCallback = () => void

export class InferenceService {
  private static instance: InferenceService
  private ws: WebSocket | null = null
  private tokenCallbacks: Set<TokenCallback> = new Set()
  private roundCallbacks: Set<RoundCallback> = new Set()
  private doneCallbacks: Set<DoneCallback> = new Set()
  private errorCallbacks: Set<ErrorCallback> = new Set()
  private store = InferenceStore.getInstance()
  private isStreaming = false

  static getInstance(): InferenceService {
    if (!InferenceService.instance) {
      InferenceService.instance = new InferenceService()
    }
    return InferenceService.instance
  }

  onToken(callback: TokenCallback): () => void {
    this.tokenCallbacks.add(callback)
    return () => this.tokenCallbacks.delete(callback)
  }

  onRound(callback: RoundCallback): () => void {
    this.roundCallbacks.add(callback)
    return () => this.roundCallbacks.delete(callback)
  }

  onDone(callback: DoneCallback): () => void {
    this.doneCallbacks.add(callback)
    return () => this.doneCallbacks.delete(callback)
  }

  onError(callback: ErrorCallback): () => void {
    this.errorCallbacks.add(callback)
    return () => this.errorCallbacks.delete(callback)
  }

  startInference(prompt: string, maxTokens: number = 64) {
    if (this.isStreaming) {
      console.warn("Inference already in progress")
      return
    }

    this.isStreaming = true
    this.store.startInference(prompt)

    // Close existing connection if any
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }

    const wsUrl = `ws://localhost:8000/api/inference/stream`
    this.ws = new WebSocket(wsUrl)

    this.ws.onopen = () => {
      console.log("WebSocket connected")
      const request = {
        prompt,
        max_tokens: maxTokens,
        draft_tokens: 8,
        temperature: 0.7,
      }
      this.ws?.send(JSON.stringify(request))
    }

    this.ws.onmessage = (event) => {
      try {
        const msg: StreamMessage = JSON.parse(event.data)

        if (msg.type === "token") {
          // Notify all token listeners
          this.tokenCallbacks.forEach((cb) => cb(msg.data))
        } else if (msg.type === "round") {
          // Notify round listeners
          const acceptanceRate = msg.data.acceptance_rate * 100
          this.roundCallbacks.forEach((cb) => cb(acceptanceRate))
        } else if (msg.type === "done") {
          // Notify done listeners
          const acceptanceRate = msg.data.acceptance_rate * 100
          this.store.markDone(acceptanceRate)
          this.doneCallbacks.forEach((cb) => cb(msg.data))
          this.isStreaming = false
        } else if (msg.type === "error") {
          console.warn("Inference error:", msg.data)
          this.errorCallbacks.forEach((cb) => cb())
          this.isStreaming = false
        }
      } catch (error) {
        console.warn("Failed to parse WebSocket message:", error)
      }
    }

    this.ws.onerror = (error) => {
      // WebSocket errors are common during tab switches, handle silently
      if (this.isStreaming) {
        console.warn("WebSocket connection interrupted")
        this.errorCallbacks.forEach((cb) => cb())
        this.isStreaming = false
      }
    }

    this.ws.onclose = (event) => {
      console.log("WebSocket closed", event.wasClean ? "(clean)" : "(unexpected)")
      this.ws = null
      if (this.isStreaming && !event.wasClean) {
        // Connection closed unexpectedly during active inference
        console.warn("WebSocket closed unexpectedly during inference")
        this.errorCallbacks.forEach((cb) => cb())
        this.isStreaming = false
      }
    }
  }

  stopInference() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.isStreaming = false
  }

  isActive(): boolean {
    return this.isStreaming
  }
}
