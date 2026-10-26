import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navigation from './components/Navigation';
import PredictPage from './pages/PredictPage';
import PipelinePage from './pages/PipelinePage';
import PerformancePage from './pages/PerformancePage';
import DatasetPage from './pages/DatasetPage';
import { checkApiHealth } from './services/predictionApi';

export default function App() {
  const [isConnected, setIsConnected] = useState(true);
  const [checkingHealth, setCheckingHealth] = useState(true);

  useEffect(() => {
    async function verifyHealth() {
      setCheckingHealth(true);
      const res = await checkApiHealth();
      setIsConnected(res.success && res.data?.model_loaded);
      setCheckingHealth(false);
    }
    verifyHealth();

    // Check health periodically every 30 seconds
    const interval = setInterval(verifyHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#F7F9FC] text-[#172B4D] py-5 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto">
          {/* Unified Navigation Header across all pages */}
          <Navigation isConnected={isConnected} checkingHealth={checkingHealth} />

          {/* Main Application Routes */}
          <main>
            <Routes>
              <Route path="/" element={<Navigate to="/predict" replace />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/pipeline" element={<PipelinePage />} />
              <Route path="/performance" element={<PerformancePage />} />
              <Route path="/dataset" element={<DatasetPage />} />
              <Route path="*" element={<Navigate to="/predict" replace />} />
            </Routes>
          </main>

          {/* Clean presentation-ready footer */}
          <footer className="mt-10 pt-5 border-t border-[#E1E7F0] text-center text-xs text-[#64748B] flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>
              Bank Marketing Machine Learning Showcase · Random Forest (Tuned Pipeline)
            </span>
            <span>
              Duration excluded for pre-contact prediction · 45,211 benchmark records
            </span>
          </footer>
        </div>
      </div>
    </BrowserRouter>
  );
}
