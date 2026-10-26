import React, { useState } from 'react';
import CustomerForm from '../components/CustomerForm';
import PredictionCard from '../components/PredictionCard';
import ModelInformation from '../components/ModelInformation';
import { predictSubscription } from '../services/predictionApi';

const DEFAULT_FORM_VALUES = {
  age: 35,
  job: 'management',
  marital: 'married',
  education: 'secondary',
  default: 'no',
  balance: 1000.0,
  housing: 'no',
  loan: 'no',
  contact: 'cellular',
  month: 'may',
  day: 15,
  campaign: 1,
  pdays: -1,
  previous: 0,
  poutcome: 'unknown',
};

export default function PredictPage() {
  const [formData, setFormData] = useState(DEFAULT_FORM_VALUES);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState(null);

  const handlePredict = async (payload) => {
    setIsLoading(true);
    setApiError(null);

    const res = await predictSubscription(payload);
    if (res.success) {
      setPredictionResult(res.data);
    } else {
      setApiError(res.error);
    }
    setIsLoading(false);
  };

  const handleReset = () => {
    setFormData(DEFAULT_FORM_VALUES);
    setPredictionResult(null);
    setApiError(null);
  };

  return (
    <div>
      {/* Page Title & Context Header */}
      <div className="mb-5">
        <h1 className="text-xl sm:text-2xl font-bold text-[#172B4D] tracking-tight m-0">
          Predict Term Deposit Subscription
        </h1>
        <p className="text-xs sm:text-sm text-[#5E6C84] mt-1">
          Input customer demographics, financial profile, and planned campaign parameters to evaluate conversion likelihood before placing a call.
        </p>
      </div>

      {/* Two-Column Responsive Workspace */}
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Left Column: Customer Form (65%) */}
        <div className="w-full lg:w-[65%] shrink-0">
          <CustomerForm
            formData={formData}
            setFormData={setFormData}
            defaultValues={DEFAULT_FORM_VALUES}
            onSubmit={handlePredict}
            onReset={handleReset}
            isLoading={isLoading}
          />
        </div>

        {/* Right Column: Prediction Results & Specifications (35%) */}
        <div className="w-full lg:w-[35%] shrink-0 lg:sticky lg:top-5">
          <PredictionCard
            result={predictionResult}
            isLoading={isLoading}
            error={apiError}
          />
          <ModelInformation />
        </div>
      </div>
    </div>
  );
}
