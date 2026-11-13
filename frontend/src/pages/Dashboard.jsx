import { useCallback, useEffect, useState } from 'react'
import { Wallet, Receipt, ShieldAlert, Ban, Clock, Percent } from 'lucide-react'
import { api } from '../api/client'
import { inr, score } from '../utils/format'
import { useApp } from '../context/AppContext'
import StatCard from '../components/StatCard'
import Ring from '../components/Ring'
import Spinner from '../components/Spinner'

export default function Dashboard() {
  const { toast } = useApp()
  const [s, setS] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(() => {
    setLoading(true)
    api.auditSummary().then(setS).catch((e) => toast(e.message, 'error')).finally(() => setLoading(false))
  }, [toast])

  useEffect(load, [load])

  if (loading) return <Spinner label="Loading dashboard…" />
  if (!s) return null

  const maxCash = Math.max(1, ...s.monthly_cashflow.map((m) => Math.max(m.sales, m.expenses)))

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Overview</h2>
          <p className="text-sm text-slate-500">Readiness, GST position and flagged items across the books.</p>
        </div>
        <button className="btn-ghost" onClick={load}>Refresh</button>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="card p-5 xl:col-span-1">
          <p className="text-sm text-slate-500">Audit readiness</p>
          <div className="mt-2 flex items-center gap-4">
            <Ring value={s.audit_readiness_score} label="score" />
            <div className="text-sm text-slate-600">
              <p><b>{s.total_invoices}</b> transactions</p>
              <p><b>{s.pending_review}</b> pending review</p>
              <p><b>{s.anomaly_count}</b> anomalies</p>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 xl:col-span-2">
          <StatCard icon={Wallet} label="Total Sales" value={inr(s.total_sales)} sub={`Output GST ${inr(s.output_gst)}`} tone="emerald" />
          <StatCard icon={Receipt} label="Total Expenses" value={inr(s.total_expenses)} sub={`Eligible ITC ${inr(s.eligible_itc)}`} tone="brand" />
          <StatCard icon={Percent} label="Net GST Payable" value={inr(s.net_gst_payable)} tone="amber" />
          <StatCard icon={Ban} label="Blocked ITC" value={s.blocked_itc_count} sub="section 17(5) items" tone="red" />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <StatCard icon={ShieldAlert} label="Critical Flags" value={s.critical_count} tone="red" />
        <StatCard icon={ShieldAlert} label="Warnings" value={s.warning_count} tone="amber" />
        <StatCard icon={Clock} label="Pending Review" value={s.pending_review} tone="slate" />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card p-5">
          <p className="font-medium">Monthly cashflow</p>
          {s.monthly_cashflow.length === 0 ? (
            <p className="mt-3 text-sm text-slate-500">No vouchers yet.</p>
          ) : (
            <div className="mt-4 space-y-3">
              {s.monthly_cashflow.map((m) => (
                <div key={m.month}>
                  <div className="mb-1 flex justify-between text-xs text-slate-500">
                    <span>{m.month}</span>
                    <span>Net {inr(m.net)}</span>
                  </div>
                  <div className="flex h-2 gap-1 overflow-hidden rounded bg-slate-100">
                    <div className="bg-emerald-500" style={{ width: `${(m.sales / maxCash) * 100}%` }} title={`Sales ${inr(m.sales)}`} />
                    <div className="bg-rose-400" style={{ width: `${(m.expenses / maxCash) * 100}%` }} title={`Expenses ${inr(m.expenses)}`} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card p-5">
          <p className="font-medium">Top categories</p>
          {s.categories.length === 0 ? (
            <p className="mt-3 text-sm text-slate-500">No data.</p>
          ) : (
            <ul className="mt-3 space-y-2 text-sm">
              {s.categories.slice(0, 8).map((c) => (
                <li key={c.category} className="flex justify-between border-b border-slate-100 pb-1">
                  <span className="truncate">{c.category}</span>
                  <span className="text-slate-500">{c.count} · {inr(c.total_amount)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="card p-5">
        <div className="flex items-center justify-between">
          <p className="font-medium">Flags by severity</p>
          <p className="text-xs text-slate-500">avg audit score {score(s.audit_readiness_score)}/100</p>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(s.flags_by_severity).length === 0
            ? <span className="text-sm text-slate-500">No open flags.</span>
            : Object.entries(s.flags_by_severity).map(([sev, n]) => (
              <span key={sev} className="rounded-full bg-slate-100 px-3 py-1 text-xs">
                <b>{sev}</b> · {n}
              </span>
            ))}
        </div>
      </div>
    </div>
  )
}