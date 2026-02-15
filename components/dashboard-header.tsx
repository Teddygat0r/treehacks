"use client"

import Link from "next/link"
import Image from "next/image"
import { usePathname } from "next/navigation"

const NAV_ITEMS = [
  { href: "/", label: "Dashboard" },
  { href: "/network", label: "Network" },
  { href: "/provider", label: "Earnings" },
  { href: "/visualize", label: "Visualize" },
]

function Nav() {
  const pathname = usePathname()
  return (
    <nav className="hidden items-center gap-1 md:flex">
      {NAV_ITEMS.map((item) => {
        const active = pathname === item.href
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              active
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
            }`}
          >
            {item.label}
          </Link>
        )
      })}
    </nav>
  )
}

export function DashboardHeader() {
  return (
    <header className="relative flex items-center border-b border-border/50 bg-card/30 px-6 py-4 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        <Image
          src="/logo.png"
          alt="Nexus logo"
          width={24}
          height={24}
          className="h-6 w-6 object-contain p-0.5"
          priority
        />
        <h1 className="font-heading text-xl font-bold tracking-tight text-foreground">Nexus</h1>
      </div>
      <div className="absolute left-1/2 -translate-x-1/2">
        <Nav />
      </div>
    </header>
  )
}
