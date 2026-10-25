import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingState() {
  return (
    <div className="bg-white border border-[#E1E7F0] rounded-xl p-8 mb-4 shadow-sm text-center">
      <div className="w-12 h-12 rounded-full bg-[#EBF3FF] text-[#2457B8] flex items-center justify-center mx-auto mb-4 animate-spin">
        <Loader2 className="w-6 h-6" />
      </div>
      <h3 className="text-base font-bold text-[#172B4D] mb-1">
        Evaluating Customer Profile
      </h3>
      <p className="text-xs text-[#5E6C84] max-w-xs mx-auto">
        Running pre-contact features through the tuned Random Forest classification pipeline...
      </p>
      <div className="w-32 h-1.5 bg-[#EDF1F7] rounded-full mx-auto mt-4 overflow-hidden">
        <div className="w-full h-full bg-[#2457B8] rounded-full animate-pulse" />
      </div>
    </div>
  );
}
