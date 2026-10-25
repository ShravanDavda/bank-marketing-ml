import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import CustomerForm from './components/CustomerForm';
import PredictionCard from './components/PredictionCard';
import ModelInformation from './components/ModelInformation';
import { checkApiHealth, predictSubscription } from './services/predictionApi';

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

export default function App() {
  const [formData, setFormData] = useState(DEFAULT_FORM_VALUES);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState(null);

  // API connectivity status
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
  }, []);

  const handlePredict = async (payload) => {
    setIsLoading(true);
    setApiError(null);

    const res = await predictSubscription(payload);
    if (res.success) {
      setPredictionResult(res.data);
      setIsConnected(true);
    } else {
      setApiError(res.error);
      setIsConnected(false);
    }
    setIsLoading(false);
  };

  const handleReset = () => {
    setFormData(DEFAULT_FORM_VALUES);
    setPredictionResult(null);
    setApiError(null);
  };

  return (
    <div className="min-h-screen bg-[#F7F9FC] text-[#172B4D] py-5 px-4 sm:px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <Header isConnected={isConnected} checkingHealth={checkingHealth} />

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
          <div className="w-full lg:w-[35%] shrink-0 sticky top-5">
            <PredictionCard
              result={predictionResult}
              isLoading={isLoading}
              error={apiError}
            />
            <ModelInformation />
          </div>
        </div>
      </div>
    </div>
  );
}
