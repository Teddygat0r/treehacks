"use client"

import { useState, useEffect } from "react"
import { Code, Copy, Check, AlertCircle } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

type Language = "python" | "javascript" | "curl"

const API_KEY_STORAGE_KEY = "nexus_api_key"

const getExamples = (apiKey: string): Record<Language, string> => ({
  python: `import requests

response = requests.post(
    "http://localhost:8000/api/inference",
    headers={
        "Authorization": "Bearer ${apiKey}"
    },
    json={
        "prompt": "Explain quantum computing",
        "max_tokens": 256
    }
)
print(response.json()["generated_text"])`,

  javascript: `const res = await fetch("http://localhost:8000/api/inference", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer ${apiKey}"
  },
  body: JSON.stringify({
    prompt: "Explain quantum computing",
    max_tokens: 256
  })
});
console.log(await res.json());`,

  curl: `curl -X POST http://localhost:8000/api/inference \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKey}" \\
  -d '{"prompt": "Explain quantum computing", "max_tokens": 256}'`,
})

export function CodeExamplesCard() {
  const [language, setLanguage] = useState<Language>("python")
  const [copied, setCopied] = useState(false)
  const [apiKey, setApiKey] = useState<string>("YOUR_API_KEY")

  // Load API key from localStorage
  useEffect(() => {
    const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY)
    if (storedKey) {
      setApiKey(storedKey)
    }

    // Listen for storage changes (when key is generated in another component)
    const handleStorageChange = () => {
      const newKey = localStorage.getItem(API_KEY_STORAGE_KEY)
      if (newKey) {
        setApiKey(newKey)
      }
    }

    window.addEventListener("storage", handleStorageChange)
    // Also listen for custom event from same tab
    window.addEventListener("nexus-api-key-updated", handleStorageChange)

    return () => {
      window.removeEventListener("storage", handleStorageChange)
      window.removeEventListener("nexus-api-key-updated", handleStorageChange)
    }
  }, [])

  const examples = getExamples(apiKey)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(examples[language])
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
      <CardContent className="p-5">
        <div className="mb-4 flex items-center gap-3">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-lg"
            style={{
              backgroundColor: "#22c55e15",
              boxShadow: "inset 0 0 0 1px #22c55e30",
            }}
          >
            <Code size={20} color="#22c55e" strokeWidth={2} />
          </div>
          <div>
            <h3 className="font-heading text-sm font-semibold text-foreground">
              Quick Start
            </h3>
            <p className="text-xs text-muted-foreground">
              Get started in seconds
            </p>
          </div>
        </div>

        <div className="mb-3 flex gap-2">
          {(["python", "javascript", "curl"] as Language[]).map((lang) => (
            <button
              key={lang}
              onClick={() => setLanguage(lang)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium capitalize transition-colors ${
                language === lang
                  ? "bg-green-500 text-white"
                  : "bg-background/50 text-muted-foreground hover:text-foreground border border-border/50"
              }`}
            >
              {lang}
            </button>
          ))}
        </div>

        <div className="relative">
          <pre className="overflow-x-auto rounded-lg border border-border/50 bg-background/80 p-3 text-xs font-mono leading-relaxed text-foreground">
            <code>{examples[language]}</code>
          </pre>
          <button
            onClick={handleCopy}
            className="absolute right-2 top-2 rounded-md bg-background/80 p-1.5 text-muted-foreground transition-colors hover:bg-background hover:text-foreground border border-border/50"
            title="Copy code"
          >
            {copied ? <Check size={12} /> : <Copy size={12} />}
          </button>
        </div>

        {apiKey === "YOUR_API_KEY" && (
          <div className="mt-3 flex items-start gap-2 rounded-md border border-blue-500/30 bg-blue-500/10 p-3">
            <AlertCircle size={14} className="mt-0.5 shrink-0 text-blue-500" />
            <p className="text-xs text-blue-400">
              Generate an API key above to get your personalized code examples
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
