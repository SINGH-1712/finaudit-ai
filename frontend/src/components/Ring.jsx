export default function Ring({ value = 0, size = 88, label }) {
    const v = Math.max(0, Math.min(100, Number(value) || 0))
    const stroke = 8, r = (size - stroke) / 2, c = 2 * Math.PI * r
    const offset = c - (v / 100) * c
    const color = v >= 80 ? '#10b981' : v >= 60 ? '#f59e0b' : '#ef4444'
    return (
      <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} stroke="#e2e8f0" strokeWidth={stroke} fill="none" />
          <circle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={stroke} fill="none"
            strokeLinecap="round" strokeDasharray={c} strokeDashoffset={offset} className="transition-all duration-500" />
        </svg>
        <div className="absolute text-center">
          <div className="text-lg font-semibold" style={{ color }}>{v.toFixed(0)}</div>
          {label && <div className="text-[10px] uppercase text-slate-500">{label}</div>}
        </div>
      </div>
    )
  }