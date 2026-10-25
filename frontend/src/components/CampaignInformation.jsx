import React from 'react';
import { Megaphone, Info } from 'lucide-react';

const CONTACT_OPTIONS = [
  { value: 'cellular', label: 'Cellular' },
  { value: 'telephone', label: 'Telephone' },
  { value: 'unknown', label: 'Unknown' },
];

const MONTH_OPTIONS = [
  { value: 'jan', label: 'January (JAN)' },
  { value: 'feb', label: 'February (FEB)' },
  { value: 'mar', label: 'March (MAR)' },
  { value: 'apr', label: 'April (APR)' },
  { value: 'may', label: 'May (MAY)' },
  { value: 'jun', label: 'June (JUN)' },
  { value: 'jul', label: 'July (JUL)' },
  { value: 'aug', label: 'August (AUG)' },
  { value: 'sep', label: 'September (SEP)' },
  { value: 'oct', label: 'October (OCT)' },
  { value: 'nov', label: 'November (NOV)' },
  { value: 'dec', label: 'December (DEC)' },
];

const POUTCOME_OPTIONS = [
  { value: 'unknown', label: 'Unknown' },
  { value: 'failure', label: 'Failure' },
  { value: 'success', label: 'Success' },
  { value: 'other', label: 'Other' },
];

export default function CampaignInformation({ formData, onChange, errors }) {
  const hasPriorContact = formData.pdays !== -1 && formData.previous > 0;

  const handlePriorContactToggle = (wasContacted) => {
    if (!wasContacted) {
      onChange('pdays', -1);
      onChange('previous', 0);
      onChange('poutcome', 'unknown');
    } else {
      onChange('pdays', formData.pdays === -1 ? 90 : formData.pdays);
      onChange('previous', formData.previous === 0 ? 1 : formData.previous);
      onChange('poutcome', formData.poutcome === 'unknown' ? 'failure' : formData.poutcome);
    }
  };

  return (
    <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 mb-4 shadow-sm">
      <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
        <div className="w-5 h-5 rounded-full bg-[#F0F4FA] text-[#2457B8] flex items-center justify-center text-xs font-bold shrink-0">
          3
        </div>
        <h2 className="text-sm sm:text-base font-bold text-[#172B4D] m-0 flex items-center gap-2">
          <Megaphone className="w-4 h-4 text-[#2457B8]" />
          Campaign Information
        </h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Contact Method */}
        <div>
          <label htmlFor="input-contact" className="form-label">
            Contact Method <span className="text-red-500">*</span>
          </label>
          <select
            id="input-contact"
            value={formData.contact}
            onChange={(e) => onChange('contact', e.target.value)}
            className="form-input-control bg-white cursor-pointer"
          >
            {CONTACT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="form-helper">Communication medium used to contact customer.</p>
        </div>

        {/* Campaign Month */}
        <div>
          <label htmlFor="input-month" className="form-label">
            Campaign Month <span className="text-red-500">*</span>
          </label>
          <select
            id="input-month"
            value={formData.month}
            onChange={(e) => onChange('month', e.target.value)}
            className="form-input-control bg-white cursor-pointer"
          >
            {MONTH_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="form-helper">Month in which marketing contact is executed.</p>
        </div>

        {/* Campaign Day */}
        <div>
          <label htmlFor="input-day" className="form-label">
            Campaign Day of Month <span className="text-red-500">*</span>
          </label>
          <input
            id="input-day"
            type="number"
            min="1"
            max="31"
            step="1"
            value={formData.day}
            onChange={(e) => onChange('day', parseInt(e.target.value, 10) || '')}
            className={`form-input-control ${errors?.day ? 'border-red-500 ring-1 ring-red-500' : ''}`}
            placeholder="1 to 31"
            required
          />
          {errors?.day ? (
            <p className="text-xs text-red-500 mt-1">{errors.day}</p>
          ) : (
            <p className="form-helper">Day of month (1 to 31).</p>
          )}
        </div>

        {/* Contacts During Current Campaign */}
        <div>
          <label htmlFor="input-campaign" className="form-label">
            Contacts During Current Campaign <span className="text-red-500">*</span>
          </label>
          <input
            id="input-campaign"
            type="number"
            min="1"
            max="65"
            step="1"
            value={formData.campaign}
            onChange={(e) => onChange('campaign', parseInt(e.target.value, 10) || '')}
            className={`form-input-control ${errors?.campaign ? 'border-red-500 ring-1 ring-red-500' : ''}`}
            placeholder="e.g. 1"
            required
          />
          {errors?.campaign ? (
            <p className="text-xs text-red-500 mt-1">{errors.campaign}</p>
          ) : (
            <p className="form-helper">Number of contacts performed in this campaign.</p>
          )}
        </div>
      </div>

      {/* Prior Contact Interaction Block */}
      <div className="mt-5 pt-4 border-t border-[#EDF1F7]">
        <label className="form-label mb-2">
          Was this customer contacted in a prior campaign? <span className="text-red-500">*</span>
        </label>
        <div className="flex items-center gap-6 mb-3">
          <label className="inline-flex items-center gap-2 cursor-pointer text-sm font-medium text-[#172B4D]">
            <input
              type="radio"
              name="prior_contact"
              checked={!hasPriorContact}
              onChange={() => handlePriorContactToggle(false)}
              className="text-[#2457B8] focus:ring-[#2457B8] w-4 h-4 cursor-pointer"
            />
            No (First-time customer)
          </label>
          <label className="inline-flex items-center gap-2 cursor-pointer text-sm font-medium text-[#172B4D]">
            <input
              type="radio"
              name="prior_contact"
              checked={hasPriorContact}
              onChange={() => handlePriorContactToggle(true)}
              className="text-[#2457B8] focus:ring-[#2457B8] w-4 h-4 cursor-pointer"
            />
            Yes (Previously contacted)
          </label>
        </div>

        {!hasPriorContact ? (
          <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3 text-xs text-[#475569] flex items-center gap-2">
            <Info className="w-4 h-4 text-[#2457B8] shrink-0" />
            <span>
              <strong>First-time contact</strong>: Sets <code className="bg-white px-1.5 py-0.5 rounded border border-[#CBD5E1] text-[#172B4D]">pdays = -1</code> and <code className="bg-white px-1.5 py-0.5 rounded border border-[#CBD5E1] text-[#172B4D]">previous = 0</code> automatically per model specification.
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-3 bg-[#F8FAFC] p-4 rounded-lg border border-[#E2E8F0]">
            <div>
              <label htmlFor="input-pdays" className="form-label text-xs">
                Days Since Previous Contact <span className="text-red-500">*</span>
              </label>
              <input
                id="input-pdays"
                type="number"
                min="0"
                max="900"
                step="1"
                value={formData.pdays}
                onChange={(e) => onChange('pdays', parseInt(e.target.value, 10) || 0)}
                className="form-input-control bg-white"
                required
              />
              <p className="form-helper">Days elapsed since last campaign contact.</p>
            </div>

            <div>
              <label htmlFor="input-previous" className="form-label text-xs">
                Previous Contacts Count <span className="text-red-500">*</span>
              </label>
              <input
                id="input-previous"
                type="number"
                min="1"
                max="100"
                step="1"
                value={formData.previous}
                onChange={(e) => onChange('previous', parseInt(e.target.value, 10) || 1)}
                className="form-input-control bg-white"
                required
              />
              <p className="form-helper">Number of contacts before this campaign.</p>
            </div>

            <div>
              <label htmlFor="input-poutcome" className="form-label text-xs">
                Outcome of Previous Campaign <span className="text-red-500">*</span>
              </label>
              <select
                id="input-poutcome"
                value={formData.poutcome}
                onChange={(e) => onChange('poutcome', e.target.value)}
                className="form-input-control bg-white cursor-pointer"
              >
                {POUTCOME_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <p className="form-helper">Result of previous marketing interaction.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
