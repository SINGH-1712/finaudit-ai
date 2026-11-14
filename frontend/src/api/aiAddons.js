import axios from 'axios'

const http = axios.create({ baseURL: '/api/ai', timeout: 90000 })

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

export const aiAddons = {
  forecast: () => http.get('/forecast').then((r) => r.data),
  retrainClassifier: (opts = {}) => http.post('/retrain-classifier', opts).then((r) => r.data),
  askLedger: (question) => http.post('/ask-ledger', { question }).then((r) => r.data),
}