import { useState } from 'react'
import { FileDown, FileSpreadsheet, FileCode2 } from 'lucide-react'
import { api } from '../api/client'
import { useApp } from '../context/AppContext'

export default function Exports() {
  const { toast } = useApp()
  const [busy, setBusy] = useState(null)

  const run = async (kind, fn, label) => {
    setBusy(kind)
    try { await fn(); toast(`${label} downloaded`, 'success') }
    catch (e) { toast(e.message, 'error') }
    finally { setBusy(null) }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold">CA Exports</h2>
        <p className="text-sm text-slate-500">Hand these files to your Chartered Accountant.</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="card p-5">
          <div className="flex items-center gap-2">
            <div className="rounded-lg bg-brand-50 p-2 text-brand-700"><FileCode2 className="h-5 w-5" /></div>
            <div>
              <p className="font-medium">Tally Prime XML</p>
              <p className="text-xs text-slate-500">Voucher-ready XML importable into Tally Prime.</p>
            </div>
          </div>
          <button className="btn-primary mt-3" disabled={busy === 'tally'} onClick={() => run('tally', api.exportTally, 'Tally XML')}>
            <FileDown className="h-4 w-4" /> {busy === 'tally' ? 'Preparing…' : 'Download Tally XML'}
          </button>
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2">
            <div className="rounded-lg bg-emerald-50 p-2 text-emerald-700"><FileSpreadsheet className="h-5 w-5" /></div>
            <div>
              <p className="font-medium">GSTR Excel Working Papers</p>
              <p className="text-xs text-slate-500">GSTR workbook including audit summary tab.</p>
            </div>
          </div>
          <button className="btn-primary mt-3" disabled={busy === 'gstr'} onClick={() => run('gstr', api.exportGst, 'GSTR Excel')}>
            <FileDown className="h-4 w-4" /> {busy === 'gstr' ? 'Preparing…' : 'Download GSTR Excel'}
          </button>
        </div>
      </div>
    </div>
  )
}