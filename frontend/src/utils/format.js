export const inr = (n) =>
    n === null || n === undefined || isNaN(n)
      ? '—'
      : new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 }).format(Number(n))
  
  export const dateOnly = (s) => {
    if (!s) return '—'
    const d = new Date(s)
    return isNaN(d) ? String(s) : d.toLocaleDateString('en-IN', { dateStyle: 'medium' })
  }
  
  export const dt = (s) => {
    if (!s) return '—'
    const d = new Date(s)
    return isNaN(d) ? String(s) : d.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
  }
  
  export const pct = (n) => (n === null || n === undefined || isNaN(n) ? '—' : `${(Number(n) * 100).toFixed(1)}%`)
  export const score = (n) => (n === null || n === undefined || isNaN(n) ? '—' : Number(n).toFixed(1))
  
  export const PAYMENT_MODES = ['CASH', 'UPI', 'CARD', 'BANK_TRANSFER', 'CHEQUE', 'CREDIT']
  
  export const RESOLUTIONS = [
    { value: 'CLARIFIED', label: 'Clarified' },
    { value: 'DISALLOW', label: 'Disallow (reject ITC)' },
    { value: 'ESCALATE_TO_CA', label: 'Escalate to CA' },
  ]