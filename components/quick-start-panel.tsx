"use client"

import { useState } from "react"
import { Copy, Check, ExternalLink } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

type Language = "python" | "javascript" | "curl"

const codeExamples: Record<Language, string> = {
  python: `import requests

# Initialize Nexus client
API_URL = "http://localhost:8000"

# Make inference request
response = requests.post(
    f"{API_URL}/api/inference",
    json={
        "prompt": "Explain quantum computing",
        "max_tokens": 256,
        "draft_tokens": 5
    }
)

result = response.json()
print(f"Generated: {result['generated_text']}")
print(f"Speedup: {result['speculation_rounds']}x")
print(f"Acceptance: {result['acceptance_rate']:.1%}")`,

  javascript: `const API_URL = "http://localhost:8000";

// Make inference request
const response = await fetch(\`\${API_URL}/api/inference\`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Explain quantum computing",
    max_tokens: 256,
    draft_tokens: 5
  })
});

const result = await response.json();
console.log(\`Generated: \${result.generated_text}\`);
console.log(\`Speedup: \${result.speculation_rounds}x\`);
console.log(\`Acceptance: \${result.acceptance_rate * 100}%\`);`,

  curl: `curl -X POST http://localhost:8000/api/inference \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 256,
    "draft_tokens": 5
  }'`,
}

const languageLabels: Record<Language, string> = {
  python: "Python",
  javascript: "JavaScript",
  curl: "cURL",
}

export function QuickStartPanel() {
  const [language, setLanguage] = useState<Language>("python")
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeExamples[language])
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm h-full">
      <CardContent className="p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="font-heading text-lg font-semibold text-foreground">
            Quick Start
          </h3>
          <a
            href="#docs"
            className="flex items-center gap-1 text-xs text-blue-500 transition-colors hover:text-blue-600"
          >
            Docs
            <ExternalLink size={12} />
          </a>
        </div>

        <div className="mb-3 flex gap-2">
          {(Object.keys(languageLabels) as Language[]).map((lang) => (
            <button
              key={lang}
              onClick={() => setLanguage(lang)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                language === lang
                  ? "bg-green-500/20 text-green-500 border border-green-500/30"
                  : "bg-background/50 text-muted-foreground hover:text-foreground border border-border/50"
              }`}
            >
              {languageLabels[lang]}
            </button>
          ))}
        </div>

        <div className="relative">
          <pre className="overflow-x-auto rounded-lg border border-border/50 bg-background/80 p-4 text-xs font-mono leading-relaxed text-foreground">
            <code>{codeExamples[language]}</code>
          </pre>
          <button
            onClick={handleCopy}
            className="absolute right-2 top-2 rounded-md bg-background/80 p-2 text-muted-foreground transition-colors hover:bg-background hover:text-foreground border border-border/50"
            title="Copy code"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
        </div>

        <div className="mt-4 rounded-md border border-blue-500/30 bg-blue-500/10 p-3">
          <p className="text-xs text-blue-400">
            <span className="font-semibold">💡 Tip:</span> Use WebSocket
            streaming for real-time token-by-token responses with{" "}
            <code className="font-mono text-blue-300">/api/inference/stream</code>
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
