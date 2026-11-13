import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ScanReceipt from './pages/ScanReceipt'
import Transactions from './pages/Transactions'
import Audit from './pages/Audit'
import Exports from './pages/Exports'
import Settings from './pages/Settings'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="scan" element={<ScanReceipt />} />
          <Route path="transactions" element={<Transactions />} />
          <Route path="audit" element={<Audit />} />
          <Route path="exports" element={<Exports />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}