import { useCallback, useEffect, useState } from 'react'
import { Search, MessageSquarePlus, AlertTriangle, Ban } from 'lucide-react'
import { api } from '../api/client'
import { useApp } from '../context/AppContext'
import { inr, dateOnly, pct, score, RESOLUTIONS } from '../utils/format'
import DataTable from '../components/DataTable'
import Badge, { statusTone } from '../components/Badge'
import EmptyState from '../components/EmptyState'
import Spinner from '../components/Spinner'
import Modal from '../components/Modal'

export default function Transactions() {
  const { toast } = useApp()
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [q, setQ] = useState('')
  const [entryType, setEntryType] = useState('')
  const [anomaliesOnly, setAnomaliesOnly] = useState(false)
  const [remarkFor, setRemarkFor] = useState(null)

  const load = useCallback(() => {
    setLoading(true)
    const params = {}
    if (q.trim()) params.q = q.trim()
    if (entryType) params.entry_type = entryType
    if (anomaliesOnly) params.anomalies_only = true
    api.listTransactions(params).then(setRows).catch((e) => toast(e.message, 'error')).finally(() => setLoading(false))
  }, [q, entryType, anomaliesOnly, toast])

  useEffect(load, [load])

  const columns = [
    { key: 'id', header: '#', render: (r) => <span className="font-mono text-xs">{r.id}</span> },
    { key: 'entry_type', header: 'Type', render: (r) => <Badge tone={r.entry_type === 'SALE' ? 'success' : 'info'}>{r.entry_type}</Badge> },
    { key: 'vendor', header: 'Party', render: (r) => (
      <div>
        <div>{r.vendor_name || '—'}</div>
        {r.vendor_gstin_status && r.vendor_gstin_status !== 'Active' && (
          <span className={`mt-0.5 inline-block rounded px-1.5 py-0.5 text-[10px] ${
            r.vendor_gstin_status === 'Cancelled' || r.vendor_gstin_status === 'Invalid'
              ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}`}>
            {r.vendor_gstin_status}
          </span>
        )}
      </div>
    ) },
    { key: 'inv', header: 'Invoice', render: (r) => <span className="font-mono text-xs">{r.invoice_no || '—'}</span> },
    { key: 'date', header: 'Date', render: (r) => dateOnly(r.invoice_date) },
    { key: 'total', header: 'Total', render: (r) => inr(r.total_amount) },
    { key: 'gst', header: 'GST', render: (r) => inr(r.gst_amount) },
    { key: 'ledger', header: 'Ledger', render: (r) => (
      <div className="max-w-[180px]">
        <p className="truncate">{r.ledger_category || '—'}</p>
        {r.ledger_confidence != null && <p className="text-xs text-slate-500">{pct(r.ledger_confidence)}</p>}
      </div>
    ) },
    { key: 'hsn', header: 'HSN/SAC', render: (r) => (
      <span className="font-mono text-xs">{r.hsn_code || '—'}</span>
    ) },
    { key: 'score', header: 'Audit', render: (r) => (
      <div className="flex items-center gap-1">
        <span>{score(r.audit_score)}</span>
        {r.is_anomaly && <AlertTriangle className="h-4 w-4 text-red-500" title="ML anomaly" />}
        {r.is_blocked_itc && <Ban className="h-4 w-4 text-amber-600" title="ITC blocked 17(5)" />}
      </div>
    ) },
    { key: 'status', header: 'Status', render: (r) => <Badge tone={statusTone(r.status)}>{r.status}</Badge> },
    { key: 'action', header: '', render: (r) => (
      <button className="btn-ghost !px-2 !py-1" title="Add remark" onClick={() => setRemarkFor(r)}>
        <MessageSquarePlus className="h-4 w-4" />
      </button>
    ) },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold">Transactions</h2>
        <p className="text-sm text-slate-500">All vouchers. Filter by party, invoice, ledger or anomalies.</p>
      </div>

      <div className="card flex flex-wrap items-end gap-3 p-4">
        <label className="flex-1 min-w-[220px]">
          <span className="label">Search</span>
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-slate-400" />
            <input className="input pl-8" value={q} onChange={(e) => setQ(e.target.value)} placeholder="party, invoice no., ledger" />
          </div>
        </label>
        <label>
          <span className="label">Entry type</span>
          <select className="input" value={entryType} onChange={(e) => setEntryType(e.target.value)}>
            <option value="">All</option>
            <option value="SALE">SALE</option>
            <option value="EXPENSE">EXPENSE</option>
          </select>
        </label>
        <label className="flex items-center gap-2 pb-2 text-sm">
          <input type="checkbox" checked={anomaliesOnly} onChange={(e) => setAnomaliesOnly(e.target.checked)} />
          Anomalies only
        </label>
      </div>

      <div className="card">
        {loading ? <Spinner /> : (
          <DataTable columns={columns} rows={rows}
            empty={<EmptyState title="No transactions" hint="Scan a receipt or seed the demo from Settings." />} />
        )}
      </div>

      <RemarkModal
        open={!!remarkFor}
        invoice={remarkFor}
        onClose={() => setRemarkFor(null)}
        onSaved={() => { setRemarkFor(null); load() }}
      />
    </div>
  )
}

export function RemarkModal({ open, invoice, flagId, onClose, onSaved }) {
  const { toast } = useApp()
  const [author, setAuthor] = useState('Advocate')
  const [remark, setRemark] = useState('')
  const [resolution, setResolution] = useState('CLARIFIED')
  const [resolvesFlag, setResolvesFlag] = useState(!!flagId)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (open) {
      setRemark(''); setResolution('CLARIFIED')
      setAuthor('Advocate'); setResolvesFlag(!!flagId)
    }
  }, [open, flagId])

  const submit = async () => {
    if (remark.trim().length < 3) return toast('Remark must be at least 3 characters.', 'error')
    setSaving(true)
    try {
      await api.saveRemark(invoice.id, {
        author, remark, resolution,
        flag_id: flagId ?? null,
        resolves_flag: resolvesFlag,
      })
      toast('Remark saved', 'success')
      onSaved?.()
    } catch (e) { toast(e.message, 'error') }
    finally { setSaving(false) }
  }

  return (
    <Modal open={open} onClose={onClose} title={`Remark · Transaction #${invoice?.id ?? ''}`}
      footer={<><button className="btn-ghost" onClick={onClose}>Cancel</button>
        <button className="btn-primary" disabled={saving} onClick={submit}>{saving ? 'Saving…' : 'Save Remark'}</button></>}>
      <div className="space-y-3">
        <label><span className="label">Author</span><input className="input" value={author} onChange={(e) => setAuthor(e.target.value)} /></label>
        <label><span className="label">Remark</span>
          <textarea rows={4} className="input" value={remark} onChange={(e) => setRemark(e.target.value)} placeholder="Describe the clarification / decision…" />
        </label>
        <label><span className="label">Resolution</span>
          <select className="input" value={resolution} onChange={(e) => setResolution(e.target.value)}>
            {RESOLUTIONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
          </select>
        </label>
        {flagId && (
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={resolvesFlag} onChange={(e) => setResolvesFlag(e.target.checked)} />
            Mark this flag as resolved
          </label>
        )}
      </div>
    </Modal>
  )
}