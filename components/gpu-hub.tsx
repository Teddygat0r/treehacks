"use client"

import { useState } from "react"
import { Plus } from "lucide-react"
import { ApiCredentialsCard } from "@/components/api-credentials-card"
import { CodeExamplesCard } from "@/components/code-examples-card"
import { TotalEarningsCard } from "@/components/total-earnings-card"
import { ConnectedHardwareCard } from "@/components/connected-hardware-card"
import { ActivityChart } from "@/components/activity-chart"

export function GpuHub() {
  const [isRegistering, setIsRegistering] = useState(false)

  const handleRegisterGpu = async () => {
    setIsRegistering(true)
    // Simulate registration
    await new Promise((resolve) => setTimeout(resolve, 1500))
    setIsRegistering(false)
    alert("GPU registered successfully! (Demo)")
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      {/* USE GPU NETWORK - Top Section */}
      <section className="mb-8">
        <h2 className="mb-2 font-heading text-2xl font-bold text-foreground">
          Use GPU Network
        </h2>
        <p className="mb-6 text-sm text-muted-foreground">
          Access distributed GPU compute for your AI inference workloads
        </p>

        <div className="grid gap-6 lg:grid-cols-2">
          <ApiCredentialsCard />
          <CodeExamplesCard />
        </div>
      </section>

      {/* PROVIDE GPU RESOURCES - Bottom Section */}
      <section>
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="mb-2 font-heading text-2xl font-bold text-foreground">
              Rent out GPU Resources
            </h2>
            <p className="text-sm text-muted-foreground">
              Monetize your idle GPU by joining the compute network
            </p>
          </div>
          <button
            onClick={handleRegisterGpu}
            disabled={isRegistering}
            className="flex items-center gap-2 rounded-md bg-green-500 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-green-600 disabled:opacity-50"
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
        </div>

        <div className="space-y-6">
          <TotalEarningsCard />
          <div className="grid gap-6 xl:grid-cols-2">
            <ConnectedHardwareCard />
            <ActivityChart />
          </div>
        </div>
      </section>
    </div>
  )
}
