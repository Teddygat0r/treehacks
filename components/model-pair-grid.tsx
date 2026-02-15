"use client"

import { useEffect, useState } from "react"
import { ModelPairCard } from "@/components/model-pair-card"
import { fetchModelPairs } from "@/lib/api"
import type { ModelPair } from "@/lib/types"

type Category = "All" | "OPT" | "Llama" | "Qwen"

export function ModelPairGrid() {
  const [pairs, setPairs] = useState<ModelPair[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedCategory, setSelectedCategory] = useState<Category>("All")

  useEffect(() => {
    const loadPairs = async () => {
      try {
        const data = await fetchModelPairs()
        setPairs(data)
      } catch (error) {
        console.error("Failed to load model pairs:", error)
      } finally {
        setLoading(false)
      }
    }

    loadPairs()
  }, [])

  const categories: Category[] = ["All", "OPT", "Llama", "Qwen"]

  const filteredPairs =
    selectedCategory === "All"
      ? pairs
      : pairs.filter((p) => p.category === selectedCategory)

  if (loading) {
    return (
      <div>
        <div className="mb-4 flex gap-2">
          {categories.map((cat) => (
            <div
              key={cat}
              className="h-9 w-20 animate-pulse rounded-md bg-card/50"
            />
          ))}
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-64 animate-pulse rounded-lg bg-card/50"
            />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-2">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              selectedCategory === cat
                ? "bg-green-500 text-white"
                : "bg-background/50 text-muted-foreground hover:text-foreground border border-border/50"
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {filteredPairs.length === 0 ? (
        <div className="flex h-64 items-center justify-center rounded-lg border border-border/50 bg-card/50 backdrop-blur-sm">
          <p className="text-sm text-muted-foreground">
            No model pairs available in this category
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {filteredPairs.map((pair, index) => (
            <ModelPairCard key={pair.id} pair={pair} index={index} />
          ))}
        </div>
      )}
    </div>
  )
}
