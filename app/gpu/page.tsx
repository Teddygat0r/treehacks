import { DashboardHeader } from "@/components/dashboard-header"
import { GpuHub } from "@/components/gpu-hub"

export const metadata = {
  title: "GPU Hub - Nexus",
  description:
    "Use the GPU network for inference or rent out your GPU to earn rewards on the Nexus network.",
}

export default function GpuPage() {
  return (
    <main className="flex h-screen flex-col overflow-hidden bg-background">
      <DashboardHeader />
      <GpuHub />
    </main>
  )
}
