import { useEffect, useRef, useState } from 'react'
import { Loader2, MessageCircle, Send, Sparkles, X } from 'lucide-react'
import { aiAddons } from '../api/aiAddons'
import { inr } from '../utils/format'
import { useApp } from '../context/AppContext'

const STARTERS = [
  'What is my net GST payable?',
  'Show blocked ITC transactions',
  'List flagged or high-value expenses',
  'How many transactions are pending review?',
]

export default function LedgerCopilot() {
  const { toast } = useApp()
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: 'Hi! Ask me anything about your books — GST payable, blocked ITC, flagged transactions, high-value expenses, duplicates.',
      suggestions: STARTERS,
    },
  ])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages, open, busy])

  const ask = async (question) => {
    const q = (question ?? input).trim()
    if (!q) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: q }])
    setBusy(true)
    try {
      const res = await aiAddons.askLedger(q)
      setMessages((m) => [...m, { role: 'assistant', text: res.answer, data: res }])
    } catch (e) {
      toast(e.message, 'error')
      setMessages((m) => [...m, { role: 'assistant', text: `Sorry — ${e.message}`, error: true }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-brand-600 px-5 py-3 text-sm font-medium text-white shadow-lg hover:bg-brand-700"
          title="Ask your ledger"
        >
          <MessageCircle className="h-4 w-4" /> Ask Ledger
        </button>
      )}

      {open && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <div className="absolute inset-0 bg-black/30" onClick={() => setOpen(false)} />
          <div className="relative flex h-full w-full max-w-md flex-col bg-white shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-brand-600" />
                <div>
                  <p className="font-semibold">Ledger Copilot</p>
                  <p className="text-xs text-slate-500">Ask plain-English questions about the books</p>
                </div>
              </div>
              <button onClick={() => setOpen(false)} className="rounded p-1 hover:bg-slate-100">
                <X className="h-4 w-4" />
              </button>
            </div>

            <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4">
              {messages.map((m, i) => <Message key={i} message={m} onSuggest={ask} />)}
              {busy && (
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <Loader2 className="h-4 w-4 animate-spin" /> Thinking…
                </div>
              )}
            </div>

            <form
              onSubmit={(e) => { e.preventDefault(); ask() }}
              className="border-t border-slate-200 p-3"
            >
              <div className="flex items-center gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="e.g. What is my net GST payable?"
                  className="input flex-1"
                  disabled={busy}
                />
                <button type="submit" disabled={busy || !input.trim()} className="btn-primary !px-3">
                  <Send className="h-4 w-4" />
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  )
}

function Message({ message, onSuggest }) {
  const isUser = message.role === 'user'
  const tone = isUser
    ? 'bg-brand-600 text-white'
    : message.error
      ? 'bg-red-50 text-red-700'
      : 'bg-slate-100 text-slate-800'

  const isMoney = (k) =>
    /gst|value|total|itc/i.test(k)

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[90%] rounded-2xl px-3 py-2 text-sm ${tone}`}>
        <p className="whitespace-pre-wrap">{message.text}</p>

        {message.data?.aggregates && Object.keys(message.data.aggregates).length > 0 && (
          <div className="mt-2 rounded bg-white/70 p-2 text-xs text-slate-700">
            {Object.entries(message.data.aggregates).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3">
                <span className="capitalize text-slate-500">{k.replace(/_/g, ' ')}</span>
                <span className="font-medium">
                  {typeof v === 'number' && isMoney(k) ? inr(v) : String(v)}
                </span>
              </div>
            ))}
          </div>
        )}

        {message.data?.matching_transactions?.length > 0 && (
          <div className="mt-2 space-y-1 rounded bg-white/70 p-2 text-xs text-slate-700">
            <p className="font-medium text-slate-500">Matching vouchers ({message.data.match_count})</p>
            {message.data.matching_transactions.slice(0, 5).map((t) => (
              <div key={t.id} className="flex items-center justify-between gap-2 border-b border-slate-100 py-1 last:border-0">
                <span className="truncate">
                  #{t.id} {t.vendor_name || '—'}{t.invoice_no ? ` · ${t.invoice_no}` : ''}
                </span>
                <span className="shrink-0 font-medium">{inr(t.total_amount)}</span>
              </div>
            ))}
            {message.data.match_count > 5 && (
              <p className="text-[10px] text-slate-500">+{message.data.match_count - 5} more</p>
            )}
          </div>
        )}

        {!isUser && message.data?.suggestions?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {message.data.suggestions.slice(0, 3).map((s, i) => (
              <button
                key={i}
                onClick={() => onSuggest(s)}
                className="rounded-full bg-white px-2 py-1 text-[11px] text-brand-700 ring-1 ring-brand-200 hover:bg-brand-50"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {!isUser && message.suggestions?.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {message.suggestions.map((s, i) => (
              <button
                key={i}
                onClick={() => onSuggest(s)}
                className="rounded-full bg-white px-2 py-1 text-[11px] text-brand-700 ring-1 ring-brand-200 hover:bg-brand-50"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}