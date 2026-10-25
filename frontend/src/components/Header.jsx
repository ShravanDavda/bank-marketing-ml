import React from 'react';
import { Landmark, Activity } from 'lucide-react';

export default function Header({ isConnected = true, checkingHealth = false }) {
  return (
    <header className="bg-white border border-[#E1E7F0] rounded-xl px-6 py-4 mb-6 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      <div className="flex items-center gap-3.5">
        <div className="w-11 h-11 rounded-lg bg-[#EBF3FF] text-[#2457B8] flex items-center justify-center shrink-0">
          <Landmark className="w-6 h-6" />
        </div>
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-[#172B4D] tracking-tight m-0">
            Bank Marketing Predictor
          </h1>
          <p className="text-xs sm:text-sm text-[#5E6C84] mt-0.5">
            Estimate the likelihood of a customer subscribing to a term deposit.
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2.5 shrink-0 self-end sm:self-auto">
        <div className="inline-flex items-center gap-2 bg-[#F8FAFC] border border-[#E2E8F0] px-3 py-1.5 rounded-md text-xs font-medium text-[#475569]">
          <span
            className={`w-2 h-2 rounded-full ${
              checkingHealth ? 'bg-amber-400 animate-pulse' : isConnected ? 'bg-[#36B37E]' : 'bg-red-500'
            }`}
          />
          <span>{checkingHealth ? 'Connecting...' : isConnected ? 'API Online' : 'API Offline'}</span>
        </div>

        <div className="hidden md:inline-flex items-center gap-1.5 bg-[#EBF3FF] text-[#2457B8] border border-[#D0E2FF] px-3 py-1.5 rounded-md text-xs font-semibold tracking-wide">
          <Activity className="w-3.5 h-3.5" />
          <span>Random Forest · Predictive Model</span>
        </div>
      </div>
    </header>
  );
}
