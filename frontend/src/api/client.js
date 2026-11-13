import axios from 'axios'

export const http = axios.create({ baseURL: '/api', timeout: 60000 })

http.interceptors.response.use(
  (r) => r,
  (err) => {
    const d = err?.response?.data?.detail
    const msg = Array.isArray(d)
      ? d.map((e) => e.msg || e.message || JSON.stringify(e)).join('; ')
      : typeof d === 'string' ? d : (err.message || 'Network error')
    return Promise.reject(new Error(msg))
  }
)

export const api = {
  // GET /api/health → { status, service }
  health: () => http.get('/health').then((r) => r.data),

  // POST /api/scan-receipt (multipart field: file) → ProcessingResult-shaped dict
  scanReceipt: (file, onProgress) => {
    const fd = new FormData()
    fd.append('file', file)
    return http.post('/scan-receipt', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => e.total && onProgress?.(Math.round((e.loaded / e.total) * 100)),
    }).then((r) => r.data)
  },

  // GET /api/transactions?q=&entry_type=&anomalies_only= → InvoiceListItem[]
  listTransactions: (params = {}) => http.get('/transactions', { params }).then((r) => r.data),

  // POST /api/transactions (body: InvoiceCreate) → InvoiceRead
  createTransaction: (body) => http.post('/transactions', body).then((r) => r.data),

  // PATCH /api/transactions/{id}/remark (body: RemarkCreate) → InvoiceRead
  saveRemark: (invoiceId, body) => http.patch(`/transactions/${invoiceId}/remark`, body).then((r) => r.data),

  // GET /api/audit-summary → AuditSummary
  auditSummary: () => http.get('/audit-summary').then((r) => r.data),

  // POST /api/seed-demo → seed report dict
  seedDemo: () => http.post('/seed-demo').then((r) => r.data),

  // GET /api/export/tally-xml → XML blob
  exportTally: async () => {
    const r = await http.get('/export/tally-xml', { responseType: 'blob' })
    _download(r.data, 'finaudit-tally-prime.xml')
  },

  // GET /api/export/gst-excel → xlsx blob
  exportGst: async () => {
    const r = await http.get('/export/gst-excel', { responseType: 'blob' })
    _download(r.data, 'finaudit-gst-working-papers.xlsx')
  },
}

function _download(blob, name) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = name
  document.body.appendChild(a); a.click(); a.remove()
  URL.revokeObjectURL(url)
}