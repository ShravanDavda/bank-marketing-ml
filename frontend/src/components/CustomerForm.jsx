import React, { useState } from 'react';
import { Send, RotateCcw } from 'lucide-react';
import PersonalProfile from './PersonalProfile';
import FinancialDetails from './FinancialDetails';
import CampaignInformation from './CampaignInformation';

export default function CustomerForm({
  formData,
  setFormData,
  defaultValues,
  onSubmit,
  onReset,
  isLoading,
}) {
  const [errors, setErrors] = useState({});

  const handleChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
    // Clear error for this field if it exists
    if (errors[field]) {
      setErrors((prev) => {
        const updated = { ...prev };
        delete updated[field];
        return updated;
      });
    }
  };

  const validate = () => {
    const errs = {};
    if (!formData.age || formData.age < 18 || formData.age > 100) {
      errs.age = 'Age must be between 18 and 100.';
    }
    if (formData.balance === '' || isNaN(formData.balance)) {
      errs.balance = 'Please enter a valid account balance.';
    }
    if (!formData.day || formData.day < 1 || formData.day > 31) {
      errs.day = 'Day must be between 1 and 31.';
    }
    if (!formData.campaign || formData.campaign < 1) {
      errs.campaign = 'Contacts during current campaign must be at least 1.';
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validate()) {
      onSubmit(formData);
    }
  };

  const handleReset = () => {
    setErrors({});
    onReset();
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      <PersonalProfile formData={formData} onChange={handleChange} errors={errors} />
      <FinancialDetails formData={formData} onChange={handleChange} errors={errors} />
      <CampaignInformation formData={formData} onChange={handleChange} errors={errors} />

      {/* Action Buttons */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-[#2457B8] hover:bg-[#1B4596] text-white font-semibold py-2.5 px-5 rounded-lg shadow-sm transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Analyzing Customer...</span>
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              <span>Predict Subscription</span>
            </>
          )}
        </button>

        <button
          type="button"
          onClick={handleReset}
          disabled={isLoading}
          className="w-full bg-white hover:bg-[#F8FAFC] text-[#475569] hover:text-[#172B4D] font-semibold py-2.5 px-5 rounded-lg border border-[#DCE3ED] hover:border-[#CBD5E1] transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Reset Form</span>
        </button>
      </div>
    </form>
  );
}
