import { useState } from 'react'
import { DatabaseZap } from 'lucide-react'
import { api } from '../api/client'
import { useApp } from '../context/AppContext'

export default function Settings() {
  const { toast } = useApp()
  const [busy, setBusy] = useState(false)
  const [report, setReport] = useState(null)

  const seed = async () => {
    if (!confirm('This wipes the ledger and reinserts the 15-voucher demo. Continue?')) return
    setBusy(true)
    try {
      const r = await api.seedDemo()
      setReport(r)
      toast(`Seeded ${r.seeded} vouchers. Readiness ${r.audit_readiness_score}/100.`, 'success')
    } catch (e) { toast(e.message, 'error') }
    finally { setBusy(false) }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold">Settings & Demo</h2>
        <p className="text-sm text-slate-500">Reset the demo dataset that exercises every statutory rule.</p>
      </div>

      <div className="card max-w-2xl p-5">
        <div className="flex items-start gap-3">
          <div className="rounded-lg bg-amber-50 p-2 text-amber-700"><DatabaseZap className="h-5 w-5" /></div>
          <div>
            <p className="font-medium">Re-seed demo vouchers</p>
            <p className="mt-1 text-sm text-slate-600">
              Wipes all invoices, flags, line items and remarks, then inserts the 15-voucher
              demonstration set that intentionally triggers every rule (cash &gt; ₹10k, ITC block,
              bad GSTIN, TDS, duplicate, missing invoice no., GST math mismatch, ML anomaly).
            </p>
            <button className="btn-primary mt-3" disabled={busy} onClick={seed}>
              {busy ? 'Seeding…' : 'Seed Demo Data'}
            </button>
          </div>
        </div>

        {report && (
          <div className="mt-5 rounded-lg bg-slate-50 p-4 text-sm">
            <p className="font-medium">Seed report</p>
            <ul className="mt-2 space-y-1 text-slate-600">
              <li>Seeded: <b>{report.seeded}</b></li>
              <li>Readiness score: <b>{report.audit_readiness_score}</b></li>
              <li>Anomalies: <b>{report.anomaly_count}</b></li>
              <li>Critical: <b>{report.critical_count}</b> · Warnings: <b>{report.warning_count}</b></li>
              <li>Rules fired: <span className="font-mono text-xs">{report.rules_fired?.join(', ')}</span></li>
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}