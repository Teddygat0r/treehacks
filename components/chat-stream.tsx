"use client"

import { motion } from "framer-motion"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { User, Bot } from "lucide-react"

type TokenType = "accepted" | "rejected" | "corrected"

interface Token {
  text: string
  type: TokenType
}

const tokenStream: Token[] = [
  { text: "The", type: "accepted" },
  { text: " theory", type: "accepted" },
  { text: " of", type: "accepted" },
  { text: " relativity,", type: "accepted" },
  { text: " proposed", type: "accepted" },
  { text: " by", type: "accepted" },
  { text: " Albert", type: "accepted" },
  { text: " Einstein", type: "accepted" },
  { text: " in", type: "accepted" },
  { text: " the early", type: "rejected" },
  { text: " 1905", type: "corrected" },
  { text: " and", type: "accepted" },
  { text: " 1915,", type: "accepted" },
  { text: " fundamentally", type: "accepted" },
  { text: " changed", type: "rejected" },
  { text: " revolutionized", type: "corrected" },
  { text: " our", type: "accepted" },
  { text: " understanding", type: "accepted" },
  { text: " of", type: "accepted" },
  { text: " space", type: "accepted" },
  { text: " and", type: "accepted" },
  { text: " time.", type: "accepted" },
  { text: " Special", type: "accepted" },
  { text: " relativity", type: "accepted" },
  { text: " shows", type: "rejected" },
  { text: " demonstrates", type: "corrected" },
  { text: " that", type: "accepted" },
  { text: " the", type: "accepted" },
  { text: " speed", type: "accepted" },
  { text: " of", type: "accepted" },
  { text: " light", type: "accepted" },
  { text: " is", type: "accepted" },
  { text: " constant", type: "accepted" },
  { text: " for", type: "accepted" },
  { text: " all", type: "accepted" },
  { text: " observers,", type: "accepted" },
  { text: " leading", type: "accepted" },
  { text: " to", type: "accepted" },
  { text: " the", type: "accepted" },
  { text: " famous", type: "rejected" },
  { text: " iconic", type: "corrected" },
  { text: " equation", type: "accepted" },
  { text: " E=mc\u00B2.", type: "accepted" },
  { text: " General", type: "accepted" },
  { text: " relativity", type: "accepted" },
  { text: " extends", type: "accepted" },
  { text: " this", type: "accepted" },
  { text: " by", type: "accepted" },
  { text: " explaining", type: "rejected" },
  { text: " describing", type: "corrected" },
  { text: " gravity", type: "accepted" },
  { text: " not", type: "accepted" },
  { text: " as", type: "accepted" },
  { text: " a", type: "accepted" },
  { text: " force,", type: "accepted" },
  { text: " but", type: "accepted" },
  { text: " as", type: "accepted" },
  { text: " a", type: "accepted" },
  { text: " curvature", type: "accepted" },
  { text: " of", type: "accepted" },
  { text: " spacetime", type: "accepted" },
  { text: " caused", type: "accepted" },
  { text: " by", type: "accepted" },
  { text: " mass", type: "rejected" },
  { text: " massive objects.", type: "corrected" },
]

function getTokenClass(type: TokenType): string {
  switch (type) {
    case "accepted":
      return "text-green-400"
    case "rejected":
      return "text-red-500 line-through opacity-70"
    case "corrected":
      return "text-blue-400 font-semibold"
  }
}

export function ChatStream() {
  return (
    <Card className="flex h-full flex-col border-border/50 bg-card/50 backdrop-blur-sm">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-sm font-medium text-muted-foreground">
          <span>Speculative Chat Stream</span>
          <div className="flex items-center gap-3 text-[10px]">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-green-400" />
              Accepted
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
              Rejected
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-blue-400" />
              Corrected
            </span>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-4 overflow-hidden">
        <ScrollArea className="flex-1 pr-4">
          <div className="flex flex-col gap-4">
            {/* User message */}
            <motion.div
              className="flex gap-3"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-secondary">
                <User className="h-4 w-4 text-foreground" />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs font-medium text-muted-foreground">You</span>
                <p className="text-sm leading-relaxed text-foreground">
                  Explain the theory of relativity.
                </p>
              </div>
            </motion.div>

            {/* AI response */}
            <motion.div
              className="flex gap-3"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.5 }}
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 ring-1 ring-primary/20">
                <Bot className="h-4 w-4 text-primary" />
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-xs font-medium text-muted-foreground">SpecNet</span>
                <div className="rounded-lg bg-secondary/50 p-3">
                  <p className="font-mono text-sm leading-relaxed">
                    {tokenStream.map((token, i) => (
                      <span key={i} className={getTokenClass(token.type)}>
                        {token.text}
                      </span>
                    ))}
                    <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-foreground" />
                  </p>
                </div>
                <div className="mt-2 flex items-center gap-4 text-[10px] text-muted-foreground">
                  <span>65 tokens drafted</span>
                  <span className="text-green-400">53 accepted</span>
                  <span className="text-red-500">6 rejected</span>
                  <span className="text-blue-400">6 corrected</span>
                </div>
              </div>
            </motion.div>
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
