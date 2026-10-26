import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  HelpCircle,
  TrendingUp,
  Percent,
  Layers,
  ShieldAlert,
} from 'lucide-react';
import { fetchPerformanceData } from '../services/predictionApi';

export default function PerformancePage() {
  const [perfData, setPerfData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      const res = await fetchPerformanceData();
      if (res.success) {
        setPerfData(res.data);
      } else {
        setError(res.error);
      }
      setIsLoading(false);
    }
    loadData();
  }, []);

  if (isLoading) {
    return (
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-12 text-center shadow-sm">
        <div className="w-8 h-8 border-3 border-[#2457B8] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-sm text-[#475569] font-medium">Loading Evaluation Results & Metrics...</p>
      </div>
    );
  }

  if (error || !perfData) {
    return (
      <div className="bg-[#FEF2F2] border border-[#FECACA] rounded-xl p-6 text-center shadow-sm">
        <p className="text-sm font-semibold text-[#B91C1C] mb-1">Failed to load Performance data</p>
        <p className="text-xs text-[#7F1D1D] mb-4">{error || 'Unknown error occurred.'}</p>
        <button
          onClick={() => window.location.reload()}
          className="text-xs bg-white text-[#B91C1C] border border-[#FECACA] px-3 py-1.5 rounded-md font-semibold hover:bg-[#FEE2E2]"
        >
          Retry
        </button>
      </div>
    );
  }

  const {
    summary_metrics = [],
    confusion_matrix: cm = {},
    baseline_vs_tuned = [],
    all_baseline_models = [],
    feature_importances = [],
    limitations = [],
  } = perfData;

  return (
    <div className="space-y-6">
      {/* 1. Page Header */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[11px] font-bold uppercase tracking-wider bg-[#DCFCE7] text-[#15803D] px-2.5 py-0.5 rounded-full border border-[#BBF7D0]">
                Verified Evaluation
              </span>
              <span className="text-[11px] text-[#64748B]">
                {perfData.test_split_ratio}
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-[#172B4D] tracking-tight m-0">
              {perfData.title}
            </h1>
            <p className="text-xs sm:text-sm text-[#5E6C84] mt-1 max-w-3xl leading-relaxed">
              {perfData.subtitle}
            </p>
          </div>

          <div className="bg-[#F8FAFC] border border-[#E2E8F0] px-3.5 py-2 rounded-lg text-xs self-start sm:self-center">
            <span className="text-[#64748B] block text-[10px] font-bold uppercase">Test Dataset</span>
            <span className="font-bold text-sm text-[#172B4D]">{perfData.test_set_size?.toLocaleString()} Records</span>
          </div>
        </div>
      </div>

      {/* 2. Key Verified Summary Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
        {summary_metrics.map((metric) => (
          <div
            key={metric.name}
            className="bg-white border border-[#E1E7F0] rounded-xl p-4 shadow-sm flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#64748B]">
                  {metric.name}
                </span>
                <span className="w-1.5 h-1.5 rounded-full bg-[#36B37E]" />
              </div>
              <div className="text-2xl sm:text-3xl font-extrabold text-[#172B4D] tracking-tight my-1">
                {metric.formatted}
              </div>
            </div>
            <div className="pt-2 mt-2 border-t border-[#F1F5F9]">
              <p className="text-[11px] text-[#475569] leading-snug">
                {metric.description}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* 3. Confusion Matrix & Decision Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: 2x2 Heatmap Matrix (7 cols) */}
        <div className="lg:col-span-7 bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
          <div className="flex items-center justify-between pb-3 mb-4 border-b border-[#EDF1F7]">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[#2457B8]" />
              <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
                Confusion Matrix (Held-out Test Set)
              </h2>
            </div>
            <span className="text-[11px] font-semibold text-[#64748B]">
              N = {cm.total_test_samples?.toLocaleString()}
            </span>
          </div>

          {/* 2x2 Grid Visualization */}
          <div className="max-w-md mx-auto my-3">
            <div className="grid grid-cols-2 gap-3 text-center">
              {/* True Negatives */}
              <div className="bg-[#F8FAFC] border-2 border-[#CBD5E1] rounded-xl p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#475569] block mb-1">
                  True Negatives (TN)
                </span>
                <span className="text-2xl sm:text-3xl font-black text-[#1E293B]">
                  {cm.true_negatives?.toLocaleString()}
                </span>
                <span className="text-[11px] text-[#64748B] block mt-1">
                  {cm.total_test_samples ? ((cm.true_negatives / cm.total_test_samples) * 100).toFixed(2) : '78.33'}% · Correct 'No'
                </span>
              </div>

              {/* False Positives */}
              <div className="bg-[#FEF2F2] border-2 border-[#FECACA] rounded-xl p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#B91C1C] block mb-1">
                  False Positives (FP)
                </span>
                <span className="text-2xl sm:text-3xl font-black text-[#B91C1C]">
                  {cm.false_positives?.toLocaleString()}
                </span>
                <span className="text-[11px] text-[#991B1B] block mt-1">
                  {cm.total_test_samples ? ((cm.false_positives / cm.total_test_samples) * 100).toFixed(2) : '9.97'}% · Called Unnecessarily
                </span>
              </div>

              {/* False Negatives */}
              <div className="bg-[#FFFBEB] border-2 border-[#FDE68A] rounded-xl p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#B45309] block mb-1">
                  False Negatives (FN)
                </span>
                <span className="text-2xl sm:text-3xl font-black text-[#B45309]">
                  {cm.false_negatives?.toLocaleString()}
                </span>
                <span className="text-[11px] text-[#92400E] block mt-1">
                  {cm.total_test_samples ? ((cm.false_negatives / cm.total_test_samples) * 100).toFixed(2) : '4.92'}% · Missed Subscribers
                </span>
              </div>

              {/* True Positives */}
              <div className="bg-[#F0FDF4] border-2 border-[#86EFAC] rounded-xl p-4">
                <span className="text-[10px] font-bold uppercase tracking-wider text-[#15803D] block mb-1">
                  True Positives (TP)
                </span>
                <span className="text-2xl sm:text-3xl font-black text-[#15803D]">
                  {cm.true_positives?.toLocaleString()}
                </span>
                <span className="text-[11px] text-[#166534] block mt-1">
                  {cm.total_test_samples ? ((cm.true_positives / cm.total_test_samples) * 100).toFixed(2) : '6.78'}% · Captured Subscribers
                </span>
              </div>
            </div>

            <div className="flex justify-between text-[11px] text-[#64748B] mt-3 px-1">
              <span>Actual Negatives: {cm.actual_no_total?.toLocaleString()} ({cm.total_test_samples ? ((cm.actual_no_total / cm.total_test_samples) * 100).toFixed(1) : '88.3'}%)</span>
              <span>Actual Positives: {cm.actual_yes_total?.toLocaleString()} ({cm.total_test_samples ? ((cm.actual_yes_total / cm.total_test_samples) * 100).toFixed(1) : '11.7'}%)</span>
            </div>
          </div>
        </div>

        {/* Right: Business Consequence Analysis (5 cols) */}
        <div className="lg:col-span-5 bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
          <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
            <HelpCircle className="w-4 h-4 text-[#2457B8]" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
              Business Error Analysis
            </h2>
          </div>

          <div className="space-y-3.5 text-xs">
            <div className="p-3 rounded-lg bg-[#FEF2F2] border border-[#FECACA]">
              <span className="font-bold text-[#B91C1C] block mb-1">
                Cost of False Positives ({cm.false_positives} leads):
              </span>
              <p className="text-[#7F1D1D] leading-relaxed">
                Calling a customer who declines incurs minor marketing advisor call time (typically 2–3 minutes). It does not create direct capital loss.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-[#FFFBEB] border border-[#FDE68A]">
              <span className="font-bold text-[#B45309] block mb-1">
                Cost of False Negatives ({cm.false_negatives} missed):
              </span>
              <p className="text-[#92400E] leading-relaxed">
                Failing to contact a customer who would have subscribed forfeits substantial deposit liquidity and interest margin. Hence, maximizing recall is prioritized over precision.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-[#F0FDF4] border border-[#BBF7D0]">
              <span className="font-bold text-[#15803D] block mb-1">
                Net Result:
              </span>
              <p className="text-[#166534] leading-relaxed">
                {cm.true_positives?.toLocaleString()} subscribers are successfully captured while eliminating {cm.true_negatives?.toLocaleString()} futile calls ({cm.actual_no_total ? ((cm.true_negatives / cm.actual_no_total) * 100).toFixed(1) : '88.7'}% of total non-subscribers avoided).
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Model Comparison Section: Baseline vs. Tuned & 4 Baseline Pipelines */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Baseline vs. Tuned Comparison (6 cols) */}
        <div className="lg:col-span-6 bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
          <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
            <TrendingUp className="w-4 h-4 text-[#2457B8]" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
              Baseline vs. Tuned Random Forest
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-[#E2E8F0] text-[#64748B]">
                  <th className="py-2 font-semibold">Model Version</th>
                  <th className="py-2 font-semibold text-right">Accuracy</th>
                  <th className="py-2 font-semibold text-right">Precision</th>
                  <th className="py-2 font-semibold text-right">Recall</th>
                  <th className="py-2 font-semibold text-right text-[#2457B8]">F1-Score</th>
                  <th className="py-2 font-semibold text-right">ROC-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F1F5F9]">
                {baseline_vs_tuned.map((row) => (
                  <tr key={row.model}>
                    <td className="py-2.5 font-bold text-[#172B4D]">
                      {row.model}
                    </td>
                    <td className="py-2.5 text-right font-mono text-[#475569]">
                      {(row.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 text-right font-mono text-[#475569]">
                      {(row.precision * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 text-right font-mono font-bold text-[#15803D]">
                      {(row.recall * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 text-right font-mono font-black text-[#2457B8]">
                      {row.f1_score.toFixed(4)}
                    </td>
                    <td className="py-2.5 text-right font-mono text-[#475569]">
                      {row.roc_auc.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-3.5 pt-3 border-t border-[#EDF1F7] text-[11px] text-[#475569] leading-relaxed">
            <strong className="text-[#172B4D]">Trade-off Interpretation: </strong>
            {baseline_vs_tuned.length >= 2 ? (
              <>
                The tuned model accepted lower precision ({((baseline_vs_tuned[0].precision || 0) * 100).toFixed(1)}% → {((baseline_vs_tuned[1].precision || 0) * 100).toFixed(1)}%) in exchange for more than doubling recall from {((baseline_vs_tuned[0].recall || 0) * 100).toFixed(1)}% to {((baseline_vs_tuned[1].recall || 0) * 100).toFixed(1)}%, raising F1-Score from {(baseline_vs_tuned[0].f1_score || 0).toFixed(4)} to {(baseline_vs_tuned[1].f1_score || 0).toFixed(4)}.
              </>
            ) : (
              "The tuned model accepted lower precision in exchange for significantly higher recall and F1-Score."
            )}
          </div>
        </div>

        {/* 4-Model Baseline Evaluation Table (6 cols) */}
        <div className="lg:col-span-6 bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
          <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
            <Layers className="w-4 h-4 text-[#2457B8]" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
              Baseline Benchmark (Section 11 & 12)
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-[#E2E8F0] text-[#64748B]">
                  <th className="py-2 font-semibold">Algorithm</th>
                  <th className="py-2 font-semibold text-right">Accuracy</th>
                  <th className="py-2 font-semibold text-right">Precision</th>
                  <th className="py-2 font-semibold text-right">Recall</th>
                  <th className="py-2 font-semibold text-right">F1-Score</th>
                  <th className="py-2 font-semibold text-right">ROC-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F1F5F9]">
                {all_baseline_models.map((m) => (
                  <tr key={m.model}>
                    <td className="py-2.5 font-semibold text-[#172B4D]">
                      {m.model}
                    </td>
                    <td className="py-2.5 text-right font-mono text-[#475569]">
                      {(m.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 text-right font-mono text-[#475569]">
                      {(m.precision * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 text-right font-mono text-[#475569]">
                      {(m.recall * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 text-right font-mono font-bold text-[#172B4D]">
                      {m.f1_score.toFixed(4)}
                    </td>
                    <td className="py-2.5 text-right font-mono text-[#475569]">
                      {m.roc_auc.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-3.5 pt-3 border-t border-[#EDF1F7] text-[11px] text-[#475569] leading-relaxed">
            Random Forest earned highest baseline F1 (0.3377); Gradient Boosting achieved highest baseline ROC-AUC (0.8023). Both were shortlisted for cross-validation tuning.
          </div>
        </div>
      </div>

      {/* 5. Feature Importance Section */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
          <div className="flex items-center gap-2">
            <Percent className="w-4 h-4 text-[#2457B8]" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
              Aggregated Feature Importance (Gini Impurity)
            </h2>
          </div>
          <span className="text-[11px] text-[#64748B]">
            Extracted directly from serialized RandomForestClassifier artifact
          </span>
        </div>

        {/* Horizontal Bars */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
          {feature_importances.map((item) => (
            <div key={item.feature} className="text-xs">
              <div className="flex justify-between items-center mb-1">
                <span className="font-semibold text-[#172B4D] capitalize">
                  {item.rank}. {item.feature.replace('_', ' ')}
                </span>
                <span className="font-mono font-bold text-[#2457B8]">
                  {item.importance_pct.toFixed(2)}%
                </span>
              </div>
              <div className="h-2 bg-[#F1F5F9] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#2457B8] rounded-full"
                  style={{ width: `${(item.importance_pct / 16) * 100}%` }}
                />
              </div>
              <p className="text-[10px] text-[#64748B] mt-0.5 truncate">
                {item.description}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* 6. Documented Limitations Section */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
          <ShieldAlert className="w-4 h-4 text-[#D97706]" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
            Documented Project Limitations (Section 16.6)
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
          {limitations.map((lim) => (
            <div
              key={lim.number}
              className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3.5 flex flex-col justify-between"
            >
              <div>
                <span className="text-[10px] font-bold uppercase text-[#D97706] block mb-1">
                  Constraint #{lim.number}
                </span>
                <h3 className="font-bold text-xs text-[#172B4D] mb-1">
                  {lim.title}
                </h3>
                <p className="text-[11px] text-[#475569] leading-relaxed">
                  {lim.detail}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
