import React from 'react';
import { DollarSign } from 'lucide-react';

export default function FinancialDetails({ formData, onChange, errors }) {
  return (
    <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 mb-4 shadow-sm">
      <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
        <div className="w-5 h-5 rounded-full bg-[#F0F4FA] text-[#2457B8] flex items-center justify-center text-xs font-bold shrink-0">
          2
        </div>
        <h2 className="text-sm sm:text-base font-bold text-[#172B4D] m-0 flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-[#2457B8]" />
          Financial Details
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Average Yearly Balance Field */}
        <div>
          <label htmlFor="input-balance" className="form-label">
            Average Yearly Balance (€) <span className="text-red-500">*</span>
          </label>
          <input
            id="input-balance"
            type="number"
            step="50"
            value={formData.balance}
            onChange={(e) => {
              const val = e.target.value;
              onChange('balance', val === '' ? '' : parseFloat(val));
            }}
            className={`form-input-control ${errors?.balance ? 'border-red-500 ring-1 ring-red-500' : ''}`}
            placeholder="e.g. 1500.00"
            required
          />
          {errors?.balance ? (
            <p className="text-xs text-red-500 mt-1">{errors.balance}</p>
          ) : (
            <p className="form-helper">Average balance in Euros. Negative balances are valid.</p>
          )}
        </div>

        {/* Credit in Default Field */}
        <div>
          <label htmlFor="input-default" className="form-label">
            Credit in Default <span className="text-red-500">*</span>
          </label>
          <select
            id="input-default"
            value={formData.default}
            onChange={(e) => onChange('default', e.target.value)}
            className="form-input-control bg-white cursor-pointer"
          >
            <option value="no">No</option>
            <option value="yes">Yes</option>
          </select>
          <p className="form-helper">Does customer currently have defaulted credit?</p>
        </div>

        {/* Housing Loan Field */}
        <div>
          <label htmlFor="input-housing" className="form-label">
            Housing Loan <span className="text-red-500">*</span>
          </label>
          <select
            id="input-housing"
            value={formData.housing}
            onChange={(e) => onChange('housing', e.target.value)}
            className="form-input-control bg-white cursor-pointer"
          >
            <option value="no">No</option>
            <option value="yes">Yes</option>
          </select>
          <p className="form-helper">Does customer currently have an active mortgage/housing loan?</p>
        </div>

        {/* Personal Loan Field */}
        <div>
          <label htmlFor="input-loan" className="form-label">
            Personal Loan <span className="text-red-500">*</span>
          </label>
          <select
            id="input-loan"
            value={formData.loan}
            onChange={(e) => onChange('loan', e.target.value)}
            className="form-input-control bg-white cursor-pointer"
          >
            <option value="no">No</option>
            <option value="yes">Yes</option>
          </select>
          <p className="form-helper">Does customer currently have an active personal loan?</p>
        </div>
      </div>
    </div>
  );
}
