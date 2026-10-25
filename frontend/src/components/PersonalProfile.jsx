import React from 'react';
import { User } from 'lucide-react';

const JOB_OPTIONS = [
  { value: 'management', label: 'Management' },
  { value: 'technician', label: 'Technician' },
  { value: 'blue-collar', label: 'Blue-Collar' },
  { value: 'admin.', label: 'Administrative' },
  { value: 'services', label: 'Services' },
  { value: 'retired', label: 'Retired' },
  { value: 'self-employed', label: 'Self-Employed' },
  { value: 'entrepreneur', label: 'Entrepreneur' },
  { value: 'student', label: 'Student' },
  { value: 'housemaid', label: 'Housemaid' },
  { value: 'unemployed', label: 'Unemployed' },
  { value: 'unknown', label: 'Unknown' },
];

const MARITAL_OPTIONS = [
  { value: 'married', label: 'Married' },
  { value: 'single', label: 'Single' },
  { value: 'divorced', label: 'Divorced / Widowed' },
];

const EDUCATION_OPTIONS = [
  { value: 'secondary', label: 'Secondary Education' },
  { value: 'tertiary', label: 'Tertiary / Higher Education' },
  { value: 'primary', label: 'Primary Education' },
  { value: 'unknown', label: 'Unknown' },
];

export default function PersonalProfile({ formData, onChange, errors }) {
  return (
    <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 mb-4 shadow-sm">
      <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
        <div className="w-5 h-5 rounded-full bg-[#F0F4FA] text-[#2457B8] flex items-center justify-center text-xs font-bold shrink-0">
          1
        </div>
        <h2 className="text-sm sm:text-base font-bold text-[#172B4D] m-0 flex items-center gap-2">
          <User className="w-4 h-4 text-[#2457B8]" />
          Personal Profile
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Age Field */}
        <div>
          <label htmlFor="input-age" className="form-label">
            Age <span className="text-red-500">*</span>
          </label>
          <input
            id="input-age"
            type="number"
            min="18"
            max="100"
            step="1"
            value={formData.age}
            onChange={(e) => onChange('age', parseInt(e.target.value, 10) || '')}
            className={`form-input-control ${errors?.age ? 'border-red-500 ring-1 ring-red-500' : ''}`}
            placeholder="e.g. 35"
            required
          />
          {errors?.age ? (
            <p className="text-xs text-red-500 mt-1">{errors.age}</p>
          ) : (
            <p className="form-helper">Customer age between 18 and 100.</p>
          )}
        </div>

        {/* Job Field */}
        <div>
          <label htmlFor="input-job" className="form-label">
            Job / Occupation <span className="text-red-500">*</span>
          </label>
          <select
            id="input-job"
            value={formData.job}
            onChange={(e) => onChange('job', e.target.value)}
            className="form-input-control bg-white cursor-pointer"
          >
            {JOB_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="form-helper">Primary professional category.</p>
        </div>

        {/* Marital Status Field */}
        <div>
          <label htmlFor="input-marital" className="form-label">
            Marital Status <span className="text-red-500">*</span>
          </label>
          <select
            id="input-marital"
            value={formData.marital}
            onChange={(e) => onChange('marital', e.target.value)}
            className="form-input-control bg-white cursor-pointer"
          >
            {MARITAL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="form-helper">Current civil or marital status.</p>
        </div>

        {/* Education Level Field */}
        <div>
          <label htmlFor="input-education" className="form-label">
            Education Level <span className="text-red-500">*</span>
          </label>
          <select
            id="input-education"
            value={formData.education}
            onChange={(e) => onChange('education', e.target.value)}
            className="form-input-control bg-white cursor-pointer"
          >
            {EDUCATION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="form-helper">Highest completed education attainment.</p>
        </div>
      </div>
    </div>
  );
}
