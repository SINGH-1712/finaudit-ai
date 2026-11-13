import { NavLink } from 'react-router-dom'
import { LayoutDashboard, ScanLine, ListChecks, ShieldAlert, Download, Settings } from 'lucide-react'

const NAV = [
  { to: '/',             label: 'Dashboard',     icon: LayoutDashboard },
  { to: '/scan',         label: 'Scan Receipt',  icon: ScanLine },
  { to: '/transactions', label: 'Transactions',  icon: ListChecks },
  { to: '/audit',        label: 'AI Audit',      icon: ShieldAlert },
  { to: '/exports',      label: 'CA Exports',    icon: Download },
  { to: '/settings',     label: 'Settings',      icon: Settings },
]

export default function Sidebar() {
  return (
    <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-white md:flex md:flex-col">
      <div className="flex items-center gap-2 px-5 py-5">
        <div className="grid h-9 w-9 place-items-center rounded-lg bg-brand-600 font-bold text-white">F</div>
        <div>
          <p className="text-sm font-semibold leading-tight">FinAudit-AI</p>
          <p className="text-xs text-slate-500">GST Bookkeeping</p>
        </div>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium ${
                isActive ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-slate-50'
              }`}>
            <Icon className="h-4 w-4" />{label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-slate-100 p-4 text-xs text-slate-500">v2.0 · B.Tech Capstone</div>
    </aside>
  )
}