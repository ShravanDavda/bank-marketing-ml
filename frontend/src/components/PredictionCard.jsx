import React from 'react';
import { BarChart3, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import LoadingState from './LoadingState';

export default function PredictionCard({ result, isLoading, error }) {
  if (isLoading) {
    return <LoadingState />;
  }

  if (error) {
    return (
      <div className="bg-[#FEF2F2] border border-[#FECACA] rounded-xl p-6 mb-4 shadow-sm">
        <div className="flex items-center gap-2.5 text-[#B91C1C] mb-2 font-bold text-sm">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>Prediction Error</span>
        </div>
        <p className="text-xs text-[#7F1D1D] mb-4 leading-relaxed">
          {error}
        </p>
        <p className="text-[11px] text-[#991B1B]">
          Please ensure the FastAPI backend is running on port 8000.
        </p>
      </div>
    );
  }

  if (!result) {
    // Initial Empty State
    return (
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-8 mb-4 shadow-sm text-center">
        <div className="w-12 h-12 rounded-full bg-[#F0F4FA] text-[#2457B8] flex items-center justify-center mx-auto mb-3.5">
          <BarChart3 className="w-6 h-6" />
        </div>
        <h3 className="text-base font-bold text-[#172B4D] mb-1.5">
          Prediction Results
        </h3>
        <p className="text-xs text-[#5E6C84] leading-relaxed max-w-xs mx-auto mb-4">
          Enter customer details and run a prediction to see the estimated subscription likelihood.
        </p>
        <div className="inline-flex items-center gap-2 bg-[#F8FAFC] border border-[#E2E8F0] px-3 py-1 rounded-md text-[11px] font-medium text-[#64748B]">
          <span className="w-1.5 h-1.5 rounded-full bg-[#36B37E]" />
          <span>Random Forest · Tuned Pipeline</span>
        </div>
      </div>
    );
  }

  // Prediction Result
  const isSubscriber = result.prediction === 'yes';
  const probPct = result.probability_available && result.subscription_probability !== null
    ? (result.subscription_probability * 100).toFixed(1)
    : null;

  return (
    <div
      className={`rounded-xl p-6 mb-4 shadow-sm border transition-all ${
        isSubscriber
          ? 'bg-[#F0FDF4] border-[#BBF7D0]'
          : 'bg-[#F8FAFC] border-[#E2E8F0]'
      }`}
    >
      {/* Top Badge & Model Row */}
      <div className="flex items-center justify-between gap-2 mb-3">
        <span
          className={`inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider px-2.5 py-1 rounded-md ${
            isSubscriber
              ? 'bg-[#DCFCE7] text-[#15803D]'
              : 'bg-[#F1F5F9] text-[#475569]'
          }`}
        >
          {isSubscriber ? (
            <CheckCircle2 className="w-3.5 h-3.5" />
          ) : (
            <XCircle className="w-3.5 h-3.5" />
          )}
          {isSubscriber ? 'Positive Outlook' : 'Standard Outlook'}
        </span>

        <span className="text-[11px] font-medium text-[#64748B]">
          {result.model_name || 'Random Forest'}
        </span>
      </div>

      {/* Outcome Headline */}
      <h2
        className={`text-xl sm:text-2xl font-bold tracking-tight mb-1 ${
          isSubscriber ? 'text-[#15803D]' : 'text-[#1E293B]'
        }`}
      >
        {result.prediction_text}
      </h2>

      {/* Probability Detail */}
      {probPct !== null && (
        <p className="text-xs sm:text-sm text-[#475569] mb-4">
          Estimated subscription probability:{' '}
          <strong className="text-[#172B4D] font-bold text-sm sm:text-base">
            {probPct}%
          </strong>
        </p>
      )}

      {/* Horizontal Probability Bar */}
      {probPct !== null && (
        <div className="my-4">
          <div className="h-2.5 bg-[#E2E8F0] rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isSubscriber ? 'bg-[#16A34A]' : 'bg-[#64748B]'
              }`}
              style={{ width: `${Math.min(Math.max(parseFloat(probPct), 2), 100)}%` }}
            />
          </div>
          <div className="flex justify-between items-center text-[11px] text-[#64748B] font-medium mt-1.5">
            <span>0%</span>
            <span className="text-[#64748B]">50% Baseline</span>
            <span>100%</span>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="pt-3.5 mt-3.5 border-t border-dashed border-[#CBD5E1] text-[11px] text-[#64748B] leading-relaxed">
        {result.disclaimer}
      </div>
    </div>
  );
}
