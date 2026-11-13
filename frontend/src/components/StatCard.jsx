const TONES = {
    brand: 'bg-brand-50 text-brand-700',
    emerald: 'bg-emerald-50 text-emerald-700',
    amber: 'bg-amber-50 text-amber-700',
    red: 'bg-red-50 text-red-700',
    slate: 'bg-slate-100 text-slate-700',
  }
  export default function StatCard({ icon: Icon, label, value, sub, tone = 'brand' }) {
    return (
      <div className="card p-5">
        <div className="flex items-start justify-between">
          <p className="text-sm text-slate-500">{label}</p>
          {Icon && <div className={`rounded-lg p-2 ${TONES[tone]}`}><Icon className="h-4 w-4" /></div>}
        </div>
        <p className="mt-2 text-2xl font-semibold tracking-tight">{value}</p>
        {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
      </div>
    )
  }