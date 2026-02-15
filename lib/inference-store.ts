/**
 * Global inference state manager using sessionStorage
 * Persists inference state across tab switches
 */

import type { VisibleToken } from "@/components/chat-panel"

export interface InferenceState {
  prompt: string
  tokens: VisibleToken[]
  done: boolean
  counts: { accepted: number; rejected: number; corrected: number; drafted: number }
  acceptanceRate: number
  isActive: boolean
  timestamp: number
}

const STORAGE_KEY = "specnet_inference_state"
const STATE_TIMEOUT_MS = 5 * 60 * 1000 // 5 minutes

export class InferenceStore {
  private static instance: InferenceStore
  private listeners: Set<(state: InferenceState | null) => void> = new Set()

  static getInstance(): InferenceStore {
    if (!InferenceStore.instance) {
      InferenceStore.instance = new InferenceStore()
    }
    return InferenceStore.instance
  }

  subscribe(listener: (state: InferenceState | null) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private notify(state: InferenceState | null) {
    this.listeners.forEach((listener) => listener(state))
  }

  getState(): InferenceState | null {
    if (typeof window === "undefined") return null

    try {
      const stored = sessionStorage.getItem(STORAGE_KEY)
      if (!stored) return null

      const state: InferenceState = JSON.parse(stored)

      // Check if state is stale (older than timeout)
      if (Date.now() - state.timestamp > STATE_TIMEOUT_MS) {
        this.clearState()
        return null
      }

      return state
    } catch (error) {
      console.error("Failed to load inference state:", error)
      return null
    }
  }

  setState(state: Partial<InferenceState>) {
    if (typeof window === "undefined") return

    try {
      const current = this.getState() || {
        prompt: "",
        tokens: [],
        done: true,
        counts: { accepted: 0, rejected: 0, corrected: 0, drafted: 0 },
        acceptanceRate: 0,
        isActive: false,
        timestamp: Date.now(),
      }

      const newState: InferenceState = {
        ...current,
        ...state,
        timestamp: Date.now(),
      }

      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(newState))
      this.notify(newState)
    } catch (error) {
      console.error("Failed to save inference state:", error)
    }
  }

  clearState() {
    if (typeof window === "undefined") return

    try {
      sessionStorage.removeItem(STORAGE_KEY)
      this.notify(null)
    } catch (error) {
      console.error("Failed to clear inference state:", error)
    }
  }

  // Update specific fields
  updateTokens(tokens: VisibleToken[]) {
    this.setState({ tokens, isActive: true })
  }

  updateCounts(counts: { accepted: number; rejected: number; corrected: number; drafted: number }) {
    this.setState({ counts })
  }

  markDone(acceptanceRate: number) {
    this.setState({ done: true, isActive: false, acceptanceRate })
  }

  startInference(prompt: string) {
    this.setState({
      prompt,
      tokens: [],
      done: false,
      counts: { accepted: 0, rejected: 0, corrected: 0, drafted: 0 },
      acceptanceRate: 0,
      isActive: true,
    })
  }
}
