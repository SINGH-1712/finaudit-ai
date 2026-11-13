import { useEffect, useState } from 'react'
import { Activity } from 'lucide-react'
import { api } from '../api/client'

export default function Topbar() {
  const [online, setOnline] = useState(null)
  useEffect(() => {
    api.health().then(() => setOnline(true)).catch(() => setOnline(false))
  }, [])
  const dot = online === true ? 'bg-emerald-500' : online === false ? 'bg-red-500' : 'bg-slate-300'
  const txt = online === null ? 'Connecting…' : online ? 'Backend online' : 'Backend offline'
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-slate-200 bg-white/80 px-6 backdrop-blur">
      <h1 className="text-sm font-medium text-slate-700">FinAudit-AI Console</h1>
      <div className="flex items-center gap-2 text-xs">
        <span className={`inline-block h-2 w-2 rounded-full ${dot}`} />
        <Activity className="h-4 w-4 text-slate-400" />
        <span className="text-slate-500">{txt}</span>
      </div>
    </header>
  )
}