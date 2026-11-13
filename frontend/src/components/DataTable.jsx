export default function DataTable({ columns, rows, rowKey = (r, i) => r.id ?? i, empty }) {
    if (!rows?.length) return empty
    return (
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              {columns.map((c) => <th key={c.key} className="whitespace-nowrap px-4 py-3 font-medium">{c.header}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={rowKey(r, i)} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/70">
                {columns.map((c) => (
                  <td key={c.key} className="px-4 py-3 align-middle">{c.render ? c.render(r) : r[c.key] ?? '—'}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }