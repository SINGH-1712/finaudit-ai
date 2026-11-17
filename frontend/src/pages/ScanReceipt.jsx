import { useRef, useState } from 'react'
import { Upload, Save, RotateCcw, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { api } from '../api/client'
import { verifyGstin } from '../api/gstin'
import { useApp } from '../context/AppContext'
import { inr, PAYMENT_MODES, pct } from '../utils/format'
import Modal from '../components/Modal'

const BLANK = {
  entry_type: 'EXPENSE', vendor_name: '', vendor_gstin: '', buyer_gstin: '',
  invoice_no: '', invoice_date: '', base_amount: '', gst_rate: '', cgst_amount: '',
  sgst_amount: '', igst_amount: '', gst_amount: '', total_amount: '', payment_mode: '',
  ledger_category: '', hsn_code: '',
}

export default function ScanReceipt() {
  const { toast } = useApp()
  const fileRef = useRef(null)
  const [preview, setPreview] = useState(null)     // scan-receipt response
  const [form, setForm] = useState(BLANK)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [saving, setSaving] = useState(false)
  const [gstinStatus, setGstinStatus] = useState(null)
  const [verifying, setVerifying] = useState(false)

  const checkGstin = async () => {
    if (!form.vendor_gstin) return
    setVerifying(true)
    try {
      const res = await verifyGstin(form.vendor_gstin)
      setGstinStatus(res)
      toast(`${res.status} · ${res.state_name || res.state_code || '—'}`, res.status === 'Active' ? 'success' : 'error')
    } catch (e) { toast(e.message, 'error') }
    finally { setVerifying(false) }
  }

  const onFile = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true); setProgress(0)
    try {
      const res = await api.scanReceipt(file, setProgress)
      setPreview(res)
      const x = res.extracted || {}
      setForm({
        ...BLANK,
        entry_type: 'EXPENSE',
        vendor_name: x.vendor_name ?? '',
        vendor_gstin: x.vendor_gstin ?? '',
        buyer_gstin: x.buyer_gstin ?? '',
        invoice_no: x.invoice_no ?? '',
        invoice_date: x.invoice_date ?? '',
        base_amount: x.base_amount ?? '',
        gst_rate: x.gst_rate ?? '',
        cgst_amount: x.cgst_amount ?? '',
        sgst_amount: x.sgst_amount ?? '',
        igst_amount: x.igst_amount ?? '',
        gst_amount: x.gst_amount ?? '',
        total_amount: x.total_amount ?? '',
        payment_mode: (x.payment_mode || '').toUpperCase(),
        ledger_category: res.ledger?.category ?? '', hsn_code: x.hsn_code ?? '',
      })
      toast('Receipt scanned — review fields before saving.', 'success')
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setUploading(false); setProgress(0)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const reset = () => { setPreview(null); setForm(BLANK) }

  const save = async () => {
    if (!preview) return
    setSaving(true)
    try {
      const num = (v) => (v === '' || v === null || v === undefined ? null : Number(v))
      const payload = {
        file_name: preview.file_name || 'receipt.jpg',
        file_path: preview.file_path,
        file_hash: preview.file_hash,
        entry_type: form.entry_type,
        vendor_name: form.vendor_name || null,
        vendor_gstin: form.vendor_gstin || null,
        buyer_gstin: form.buyer_gstin || null,
        invoice_no: form.invoice_no || null,
        invoice_date: form.invoice_date || null,
        base_amount: num(form.base_amount),
        gst_rate: num(form.gst_rate),
        cgst_amount: num(form.cgst_amount),
        sgst_amount: num(form.sgst_amount),
        igst_amount: num(form.igst_amount),
        gst_amount: num(form.gst_amount),
        total_amount: num(form.total_amount),
        payment_mode: form.payment_mode || null,
        hsn_code: form.hsn_code || null,
        ledger_category: form.ledger_category || null,
        raw_text: preview.raw_text || null,
        line_items: (preview.line_items || []).map((li) => ({
          description: li.description,
          hsn_sac: li.hsn_sac ?? null,
          quantity: li.quantity ?? null,
          rate: li.rate ?? null,
          amount: li.amount ?? null,
        })),
        ocr_engine: preview.ocr?.engine ?? null,
        ocr_mean_confidence: preview.ocr?.mean_confidence ?? null,
        ocr_token_count: preview.ocr?.token_count ?? null,
        field_confidences: preview.field_confidences || null,
        field_sources: preview.field_sources || null,
        preprocess_meta: preview.ocr?.preprocess || null,
      }
      const inv = await api.createTransaction(payload)
      toast(`Saved transaction #${inv.id}`, 'success')
      reset()
    } catch (e) {
      toast(e.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Scan Receipt</h2>
          <p className="text-sm text-slate-500">Upload a bill image — OCR + Isolation Forest + ledger classifier run together.</p>
        </div>
        <div className="flex gap-2">
          <input ref={fileRef} type="file" accept=".png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp" hidden onChange={onFile} />
          <button className="btn-primary" disabled={uploading} onClick={() => fileRef.current?.click()}>
            <Upload className="h-4 w-4" /> {uploading ? `Scanning ${progress}%…` : 'Upload Receipt'}
          </button>
          {preview && <button className="btn-ghost" onClick={reset}><RotateCcw className="h-4 w-4" /> Reset</button>}
        </div>
      </div>

      {uploading && (
        <div className="card p-4">
          <div className="mb-1 flex justify-between text-xs text-slate-500">
            <span>Uploading & OCR processing…</span><span>{progress}%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
            <div className="h-full bg-brand-600 transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {!preview && !uploading && (
        <div className="card p-10 text-center">
          <Upload className="mx-auto h-10 w-10 text-slate-300" />
          <p className="mt-3 font-medium text-slate-700">No receipt loaded</p>
          <p className="text-sm text-slate-500">Upload a PNG/JPG/WEBP/TIFF/BMP up to 12 MB.</p>
        </div>
      )}

      {preview && (
        <ScanReview preview={preview} form={form} setForm={setForm} onSave={save} saving={saving} />
      )}
    </div>
  )
}

function ScanReview({ preview, form, setForm, onSave, saving }) {
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })
  const ocr = preview.ocr || {}
  const ledger = preview.ledger
  const anomaly = preview.anomaly

  const [gstinStatus, setGstinStatus] = useState(null)
  const [verifying, setVerifying] = useState(false)

  const checkGstin = async () => {
    if (!form.vendor_gstin) return
    setVerifying(true)
    try {
      const res = await api.verifyGstin(form.vendor_gstin)
      setGstinStatus(res)
    } catch (e) {
      console.error(e)
    } finally {
      setVerifying(false)
    }
  }
  

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-3">
        <Mini label="OCR quality" value={`${pct(ocr.mean_confidence)} mean · ${ocr.token_count ?? 0} tokens`} good={ocr.mean_confidence >= 0.7} />
        <Mini
          label="Ledger classifier"
          value={ledger ? `${ledger.category} · ${pct(ledger.confidence)}` : '—'}
          good={ledger && ledger.confidence >= 0.6}
          note={ledger?.label}
        />
        <Mini
          label="Anomaly detector"
          value={anomaly ? (anomaly.is_anomaly ? `ANOMALY · score ${(anomaly.score * 100).toFixed(1)}` : `Normal · score ${(anomaly.score * 100).toFixed(1)}`) : '—'}
          good={anomaly && !anomaly.is_anomaly}
        />
      </div>

      {!!preview.warnings?.length && (
        <div className="card border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          <p className="flex items-center gap-2 font-medium"><AlertTriangle className="h-4 w-4" /> OCR warnings</p>
          <ul className="ml-6 mt-1 list-disc space-y-1">
            {preview.warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      <div className="card p-5">
        <h3 className="mb-3 font-semibold">Confirm extracted fields</h3>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Entry type">
            <select className="input" value={form.entry_type} onChange={set('entry_type')}>
              <option value="EXPENSE">EXPENSE</option>
              <option value="SALE">SALE</option>
            </select>
          </Field>
          <Field label="Vendor name"><input className="input" value={form.vendor_name} onChange={set('vendor_name')} /></Field>
          <Field label="Vendor GSTIN">
            <div className="flex gap-2">
              <input className="input font-mono flex-1" value={form.vendor_gstin} onChange={set('vendor_gstin')} />
              <button type="button" className="btn-ghost !px-2 !py-1 text-xs"
                disabled={!form.vendor_gstin || verifying} onClick={checkGstin}>
                {verifying ? '…' : 'Verify'}
              </button>
            </div>
            {gstinStatus && (
              <div className={`mt-1 rounded px-2 py-1 text-xs ${
                gstinStatus.status === 'Active' ? 'bg-emerald-50 text-emerald-700'
                : gstinStatus.status === 'Cancelled' || gstinStatus.status === 'Invalid' ? 'bg-red-50 text-red-700'
                : 'bg-amber-50 text-amber-700'}`}>
                <b>{gstinStatus.status}</b>
                {gstinStatus.state_name && ` · ${gstinStatus.state_name}`}
                {gstinStatus.entity_type && ` · ${gstinStatus.entity_type}`}
              </div>
            )}
          </Field>
          <Field label="Buyer GSTIN"><input className="input font-mono" value={form.buyer_gstin} onChange={set('buyer_gstin')} /></Field>
          <Field label="Invoice No."><input className="input" value={form.invoice_no} onChange={set('invoice_no')} /></Field>
          <Field label="Invoice date"><input type="date" className="input" value={form.invoice_date || ''} onChange={set('invoice_date')} /></Field>
          <Field label="Base amount (₹)"><input type="number" step="0.01" className="input" value={form.base_amount} onChange={set('base_amount')} /></Field>
          <Field label="GST rate (%)"><input type="number" step="0.01" className="input" value={form.gst_rate} onChange={set('gst_rate')} /></Field>
          <Field label="CGST (₹)"><input type="number" step="0.01" className="input" value={form.cgst_amount} onChange={set('cgst_amount')} /></Field>
          <Field label="SGST (₹)"><input type="number" step="0.01" className="input" value={form.sgst_amount} onChange={set('sgst_amount')} /></Field>
          <Field label="IGST (₹)"><input type="number" step="0.01" className="input" value={form.igst_amount} onChange={set('igst_amount')} /></Field>
          <Field label="Total GST (₹)"><input type="number" step="0.01" className="input" value={form.gst_amount} onChange={set('gst_amount')} /></Field>
          <Field label="Total (₹)"><input type="number" step="0.01" className="input" value={form.total_amount} onChange={set('total_amount')} /></Field>
          <Field label="Payment mode">
            <select className="input" value={form.payment_mode} onChange={set('payment_mode')}>
              <option value="">—</option>
              {PAYMENT_MODES.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Ledger category override">
            <input className="input" value={form.ledger_category} onChange={set('ledger_category')} placeholder="Leave blank to keep ML label" />
          </Field>
          <Field label="HSN / SAC Code">
            <input className="input font-mono" value={form.hsn_code}
              onChange={set('hsn_code')} placeholder="4, 6 or 8 digits (e.g. 9983)" />
          </Field>
        </div>
      </div>

      {!!preview.line_items?.length && (
        <div className="card p-5">
          <h3 className="mb-3 font-semibold">Line items</h3>
          <ul className="divide-y divide-slate-100 text-sm">
            {preview.line_items.map((li, i) => (
              <li key={i} className="flex justify-between py-2">
                <span className="truncate">{li.description}</span>
                <span className="text-slate-500">{inr(li.amount)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {!!preview.raw_text && (
        <details className="card p-5 text-sm">
          <summary className="cursor-pointer font-medium">Raw OCR text</summary>
          <pre className="mt-2 whitespace-pre-wrap text-xs text-slate-600">{preview.raw_text}</pre>
        </details>
      )}

      <div className="flex justify-end">
        <button className="btn-primary" disabled={saving} onClick={onSave}>
          <Save className="h-4 w-4" /> {saving ? 'Saving…' : 'Save Transaction'}
        </button>
      </div>
    </div>
  )
}

function Field({ label, children }) {
  return <label><span className="label">{label}</span>{children}</label>
}
function Mini({ label, value, note, good }) {
  return (
    <div className="card p-4">
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-1 flex items-center gap-1 text-sm font-medium ${good === false ? 'text-red-600' : good === true ? 'text-emerald-700' : ''}`}>
        {good === true && <CheckCircle2 className="h-4 w-4" />}
        {good === false && <AlertTriangle className="h-4 w-4" />}
        {value}
      </p>
      {note && <p className="mt-0.5 truncate text-xs text-slate-500">{note}</p>}
    </div>
  )
}