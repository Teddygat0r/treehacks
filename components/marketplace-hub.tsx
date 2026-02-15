"use client"

import { ChevronDown } from "lucide-react"
import { MarketplaceStats } from "@/components/marketplace-stats"
import { ModelPairGrid } from "@/components/model-pair-grid"
import { QuickStartPanel } from "@/components/quick-start-panel"
import { ProviderDashboard } from "@/components/provider-dashboard"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"

export function MarketplaceHub() {
  return (
    <div className="flex-1 overflow-y-auto p-6">
      {/* Hero Stats */}
      <section className="mb-6">
        <h2 className="mb-4 font-heading text-2xl font-bold text-foreground">
          Marketplace
        </h2>
        <MarketplaceStats />
      </section>

      {/* Two-column: Models + Quick Start */}
      <div className="grid gap-6 xl:grid-cols-2 mb-6">
        <section>
          <h3 className="mb-4 font-heading text-lg font-semibold text-foreground">
            Available Model Pairs
          </h3>
          <ModelPairGrid />
        </section>
        <section>
          <h3 className="mb-4 font-heading text-lg font-semibold text-foreground">
            Get Started
          </h3>
          <QuickStartPanel />
        </section>
      </div>

      {/* Provider Earnings (Collapsible) */}
      <Collapsible defaultOpen>
        <CollapsibleTrigger className="group flex w-full items-center justify-between rounded-lg border border-border/50 bg-card/30 px-4 py-3 transition-colors hover:bg-card/50 mb-4">
          <h3 className="font-heading text-lg font-semibold text-foreground">
            Provider Earnings
          </h3>
          <ChevronDown
            size={20}
            className="text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180"
          />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <ProviderDashboard />
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}
