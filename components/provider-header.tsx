"use client"

import { motion } from "framer-motion"
export function ProviderHeader() {
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h2 className="font-heading text-2xl font-bold tracking-tight text-foreground">
        Earnings Dashboard
      </h2>
      <p className="mt-1 text-sm text-muted-foreground">
        Manage your Draft Node and track earnings.
      </p>
    </motion.div>
  )
}
