import { AlertTriangle, CheckCircle2, Info } from 'lucide-react'
import { useApp } from '../context/AppContext'

export default function Toasts() {
  const { toasts } = useApp()
  const ICONS = { success: CheckCircle2, error: AlertTriangle, info: Info }
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
      {toasts.map((t) => {
        const Icon = ICONS[t.type] || Info
        return (
          <div key={t.id} className="card flex items-center gap-2 px-4 py-3 text-sm shadow-lg">
            <Icon className={`h-4 w-4 ${t.type === 'error' ? 'text-red-600' : 'text-brand-600'}`} />
            <span className="max-w-sm">{t.message}</span>
          </div>
        )
      })}
    </div>
  )
}