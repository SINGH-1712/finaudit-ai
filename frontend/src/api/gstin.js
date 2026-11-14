import axios from 'axios'

const http = axios.create({ baseURL: '', timeout: 15000 })

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

export const verifyGstin = (gstin) =>
  http.get(`/api/verify-gstin/${encodeURIComponent(gstin)}`).then((r) => r.data)