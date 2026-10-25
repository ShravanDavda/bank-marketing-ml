import React from 'react';
import { Settings, ShieldCheck } from 'lucide-react';

export default function ModelInformation() {
  return (
    <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 shadow-sm">
      <div className="flex items-center gap-2 pb-2.5 mb-3 border-b border-[#EDF1F7]">
        <Settings className="w-4 h-4 text-[#2457B8]" />
        <h3 className="text-xs font-bold uppercase tracking-wider text-[#172B4D] m-0">
          Model Specifications
        </h3>
      </div>

      <div className="space-y-2 text-xs">
        <div className="flex justify-between items-center py-1">
          <span className="text-[#64748B]">Model Architecture</span>
          <span className="font-semibold text-[#172B4D]">Random Forest (Tuned)</span>
        </div>

        <div className="flex justify-between items-center py-1 border-t border-[#F1F5F9]">
          <span className="text-[#64748B]">Prediction Task</span>
          <span className="font-semibold text-[#172B4D]">Binary Classification</span>
        </div>

        <div className="flex justify-between items-center py-1 border-t border-[#F1F5F9]">
          <span className="text-[#64748B]">Target Variable</span>
          <span className="font-semibold text-[#172B4D]">Term Deposit (yes / no)</span>
        </div>

        <div className="flex justify-between items-center py-1 border-t border-[#F1F5F9]">
          <span className="text-[#64748B]">Timing Constraint</span>
          <span className="font-semibold text-[#172B4D]">Pre-contact (duration excluded)</span>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-[#EDF1F7] flex items-center gap-1.5 text-[11px] text-[#059669]">
        <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
        <span>Fitted preprocessing pipeline active</span>
      </div>
    </div>
  );
}
