import { useCallback, useEffect, useMemo, useState } from 'react'
import { ShieldAlert, ShieldCheck, Filter } from 'lucide-react'
import { api } from '../api/client'
import { useApp } from '../context/AppContext'
import { inr, dateOnly, pct, score } from '../utils/format'
import Badge, { severityTone } from '../components/Badge'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'
import Ring from '../components/Ring'
import { RemarkModal } from './Transactions'

export default function Audit() {
  const { toast } = useApp()
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [severity, setSeverity] = useState('')
  const [source, setSource] = useState('')
  const [showResolved, setShowResolved] = useState(false)
  const [target, setTarget] = useState(null)

  const load = useCallback(() => {
    setLoading(true)
    api.auditSummary().then(setSummary).catch((e) => toast(e.message, 'error')).finally(() => setLoading(false))
  }, [toast])

  useEffect(load, [load])

  const issues = useMemo(() => {
    if (!summary) return []
    return summary.issues.filter((i) => {
      if (!showResolved && i.is_resolved) return false
      if (severity && i.severity !== severity) return false
      if (source && i.source !== source) return false
      return true
    })
  }, [summary, severity, source, showResolved])

  if (loading) return <Spinner label="Loading audit…" />
  if (!summary) return null

  const sevOptions = Array.from(new Set(summary.issues.map((i) => i.severity)))
  const srcOptions = Array.from(new Set(summary.issues.map((i) => i.source)))

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">AI Audit</h2>
          <p className="text-sm text-slate-500">Hybrid rule engine + Isolation Forest flags. Add remarks to resolve.</p>
        </div>
        <button className="btn-ghost" onClick={load}>Refresh</button>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="card flex items-center gap-4 p-5 md:col-span-1">
          <Ring value={summary.audit_readiness_score} label="readiness" />
          <div className="text-sm text-slate-600">
            <p>Total issues: <b>{summary.issues.length}</b></p>
            <p>Critical: <b>{summary.critical_count}</b></p>
            <p>Warnings: <b>{summary.warning_count}</b></p>
          </div>
        </div>
        <div className="card p-5">
          <p className="text-xs uppercase text-slate-500">Blocked ITC</p>
          <p className="mt-1 text-2xl font-semibold">{summary.blocked_itc_count}</p>
          <p className="text-xs text-slate-500">section 17(5) items</p>
        </div>
        <div className="card p-5">
          <p className="text-xs uppercase text-slate-500">Anomalies</p>
          <p className="mt-1 text-2xl font-semibold">{summary.anomaly_count}</p>
          <p className="text-xs text-slate-500">ML flagged</p>
        </div>
        <div className="card p-5">
          <p className="text-xs uppercase text-slate-500">Pending review</p>
          <p className="mt-1 text-2xl font-semibold">{summary.pending_review}</p>
          <p className="text-xs text-slate-500">awaiting advocate</p>
        </div>
      </div>

      <div className="card flex flex-wrap items-center gap-3 p-4">
        <span className="flex items-center gap-1 text-sm text-slate-600"><Filter className="h-4 w-4" /> Filters</span>
        <select className="input !w-auto" value={severity} onChange={(e) => setSeverity(e.target.value)}>
          <option value="">All severities</option>
          {sevOptions.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select className="input !w-auto" value={source} onChange={(e) => setSource(e.target.value)}>
          <option value="">All sources</option>
          {srcOptions.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={showResolved} onChange={(e) => setShowResolved(e.target.checked)} /> Show resolved
        </label>
        <span className="ml-auto text-xs text-slate-500">{issues.length} of {summary.issues.length}</span>
      </div>

      {issues.length === 0 ? (
        <div className="card"><EmptyState title="No issues match filters" hint="Try clearing filters or run the demo seed in Settings." /></div>
      ) : (
        <ul className="space-y-3">
          {issues.map((i) => (
            <li key={i.id} className="card p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone={severityTone(i.severity)}>{i.severity}</Badge>
                    <Badge tone={i.source === 'ML' ? 'info' : 'warning'}>{i.source}</Badge>
                    <span className="font-mono text-xs text-slate-500">{i.rule_code}</span>
                    {i.is_resolved && <Badge tone="success"><ShieldCheck className="mr-1 h-3 w-3" />Resolved</Badge>}
                  </div>
                  <p className="mt-2 font-medium">{i.title}</p>
                  <p className="text-sm text-slate-600">{i.message}</p>
                  {i.legal_reference && <p className="mt-1 text-xs text-slate-500">Ref: {i.legal_reference}</p>}
                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                    <span>Tx #{i.invoice_id}</span>
                    {i.vendor_name && <span>{i.vendor_name}</span>}
                    {i.invoice_no && <span>Inv {i.invoice_no}</span>}
                    {i.invoice_date && <span>{dateOnly(i.invoice_date)}</span>}
                    {i.total_amount != null && <span>{inr(i.total_amount)}</span>}
                    {i.ledger_category && <span>{i.ledger_category} · {pct(i.ml_confidence)}</span>}
                    {i.anomaly_score != null && <span>anomaly {score(i.anomaly_score)}</span>}
                    {i.penalty > 0 && <span className="text-red-600">penalty {inr(i.penalty)}</span>}
                  </div>
                  {i.advocate_remark && (
                    <p className="mt-2 rounded bg-slate-50 p-2 text-xs italic text-slate-600">“{i.advocate_remark}”</p>
                  )}
                </div>
                {!i.is_resolved && (
                  <button className="btn-primary !py-1.5" onClick={() => setTarget(i)}>
                    <ShieldAlert className="h-4 w-4" /> Add remark / Resolve
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      <RemarkModal
        open={!!target}
        invoice={target ? { id: target.invoice_id } : null}
        flagId={target?.id}
        onClose={() => setTarget(null)}
        onSaved={() => { setTarget(null); load() }}
      />
    </div>
  )
}