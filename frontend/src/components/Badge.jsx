const T = {
    success: 'bg-emerald-100 text-emerald-700',
    warning: 'bg-amber-100 text-amber-700',
    danger: 'bg-red-100 text-red-700',
    info: 'bg-blue-100 text-blue-700',
    neutral: 'bg-slate-100 text-slate-600',
  }
  export default function Badge({ tone = 'neutral', children }) {
    return <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${T[tone] || T.neutral}`}>{children}</span>
  }
  export function severityTone(sev = '') {
    const s = String(sev).toUpperCase()
    if (s === 'CRITICAL') return 'danger'
    if (s === 'WARNING' || s === 'HIGH') return 'warning'
    return 'info'
  }
  export function statusTone(status = '') {
    const s = String(status).toUpperCase()
    if (s.includes('DISALLOW')) return 'danger'
    if (s.includes('ESCALAT')) return 'warning'
    if (s.includes('CLARIF') || s.includes('REVIEW')) return 'success'
    return 'neutral'
  }