import React, { useState, useEffect } from 'react';
import {
  Database,
  Search,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  XCircle,
  AlertCircle,
  PieChart,
  BarChart2,
  Table as TableIcon,
} from 'lucide-react';
import {
  fetchDatasetSummary,
  fetchDatasetFeatures,
  fetchDatasetPreview,
  fetchDatasetEDA,
} from '../services/predictionApi';

export default function DatasetPage() {
  const [summary, setSummary] = useState(null);
  const [features, setFeatures] = useState([]);
  const [previewData, setPreviewData] = useState(null);
  const [edaData, setEdaData] = useState(null);

  const [previewPage, setPreviewPage] = useState(1);
  const [featureSearch, setFeatureSearch] = useState('');
  const [featureFilter, setFeatureFilter] = useState('all');

  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [error, setError] = useState(null);

  // Load summary, features, and EDA on mount
  useEffect(() => {
    async function loadInitialData() {
      setIsLoading(true);
      const [sumRes, featRes, edaRes] = await Promise.all([
        fetchDatasetSummary(),
        fetchDatasetFeatures(),
        fetchDatasetEDA(),
      ]);

      if (sumRes.success && featRes.success) {
        setSummary(sumRes.data);
        setFeatures(featRes.data.features || []);
        if (edaRes.success) {
          setEdaData(edaRes.data);
        }
      } else {
        setError(sumRes.error || featRes.error || 'Failed to load dataset details.');
      }
      setIsLoading(false);
    }
    loadInitialData();
  }, []);

  // Load preview data when page changes
  useEffect(() => {
    async function loadPreview() {
      setIsLoadingPreview(true);
      const res = await fetchDatasetPreview(previewPage, 10);
      if (res.success) {
        setPreviewData(res.data);
      }
      setIsLoadingPreview(false);
    }
    loadPreview();
  }, [previewPage]);

  if (isLoading) {
    return (
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-12 text-center shadow-sm">
        <div className="w-8 h-8 border-3 border-[#2457B8] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-sm text-[#475569] font-medium">Loading Dataset Explorer & Statistics...</p>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="bg-[#FEF2F2] border border-[#FECACA] rounded-xl p-6 text-center shadow-sm">
        <p className="text-sm font-semibold text-[#B91C1C] mb-1">Failed to load Dataset</p>
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

  // Filter features
  const filteredFeatures = features.filter((f) => {
    const matchesSearch =
      f.name.toLowerCase().includes(featureSearch.toLowerCase()) ||
      f.description.toLowerCase().includes(featureSearch.toLowerCase());

    if (!matchesSearch) return false;

    if (featureFilter === 'numerical') return f.data_type.includes('int') || f.data_type.includes('num');
    if (featureFilter === 'categorical') return f.data_type.includes('cat') || f.data_type.includes('bin');
    if (featureFilter === 'inputs') return f.used_in_model;
    if (featureFilter === 'excluded') return !f.used_in_model;
    return true;
  });

  return (
    <div className="space-y-6">
      {/* 1. Page Header */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[11px] font-bold uppercase tracking-wider bg-[#EBF3FF] text-[#2457B8] px-2.5 py-0.5 rounded-full border border-[#D0E2FF]">
                Source of Truth
              </span>
              <span className="text-[11px] text-[#64748B]">
                data/bank-full.csv · Read-Only Access
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-[#172B4D] tracking-tight m-0">
              Dataset Explorer & Descriptive Statistics
            </h1>
            <p className="text-xs sm:text-sm text-[#5E6C84] mt-1 max-w-3xl leading-relaxed">
              Explore the original Portuguese banking marketing dataset, feature definitions, target imbalance, and descriptive associations.
            </p>
          </div>

          <div className="bg-[#F8FAFC] border border-[#E2E8F0] px-3.5 py-2 rounded-lg text-xs self-start sm:self-center shrink-0">
            <span className="text-[#64748B] block text-[10px] font-bold uppercase">Delimiter</span>
            <span className="font-mono font-bold text-sm text-[#172B4D]">Semicolon ( ; )</span>
          </div>
        </div>
      </div>

      {/* 2. Top-Level Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
        <div className="bg-white border border-[#E1E7F0] rounded-xl p-4 shadow-sm">
          <span className="text-[11px] font-bold uppercase text-[#64748B] block mb-1">Total Records</span>
          <div className="text-xl sm:text-2xl font-extrabold text-[#172B4D]">
            {summary.total_records.toLocaleString()}
          </div>
          <span className="text-[10px] text-[#36B37E] font-medium block mt-1">100% complete</span>
        </div>

        <div className="bg-white border border-[#E1E7F0] rounded-xl p-4 shadow-sm">
          <span className="text-[11px] font-bold uppercase text-[#64748B] block mb-1">Columns</span>
          <div className="text-xl sm:text-2xl font-extrabold text-[#172B4D]">
            {summary.total_columns}
          </div>
          <span className="text-[10px] text-[#64748B] block mt-1">16 features + 1 target</span>
        </div>

        <div className="bg-white border border-[#E1E7F0] rounded-xl p-4 shadow-sm">
          <span className="text-[11px] font-bold uppercase text-[#64748B] block mb-1">Missing Values</span>
          <div className="text-xl sm:text-2xl font-extrabold text-[#16A34A]">
            0
          </div>
          <span className="text-[10px] text-[#166534] block mt-1">No imputation needed</span>
        </div>

        <div className="bg-white border border-[#E1E7F0] rounded-xl p-4 shadow-sm">
          <span className="text-[11px] font-bold uppercase text-[#64748B] block mb-1">Duplicates</span>
          <div className="text-xl sm:text-2xl font-extrabold text-[#16A34A]">
            0
          </div>
          <span className="text-[10px] text-[#166534] block mt-1">Zero row deletions</span>
        </div>

        <div className="bg-white border border-[#E1E7F0] rounded-xl p-4 shadow-sm">
          <span className="text-[11px] font-bold uppercase text-[#64748B] block mb-1">Subscribers (yes)</span>
          <div className="text-xl sm:text-2xl font-extrabold text-[#2457B8]">
            {summary.target_distribution?.yes?.count.toLocaleString()}
          </div>
          <span className="text-[10px] text-[#2457B8] font-bold block mt-1">
            {summary.target_distribution?.yes?.percentage}% minority
          </span>
        </div>

        <div className="bg-white border border-[#E1E7F0] rounded-xl p-4 shadow-sm">
          <span className="text-[11px] font-bold uppercase text-[#64748B] block mb-1">Non-Subscribers</span>
          <div className="text-xl sm:text-2xl font-extrabold text-[#475569]">
            {summary.target_distribution?.no?.count.toLocaleString()}
          </div>
          <span className="text-[10px] text-[#64748B] block mt-1">
            {summary.target_distribution?.no?.percentage}% majority
          </span>
        </div>
      </div>

      {/* 3. Target Distribution Section */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
          <PieChart className="w-4 h-4 text-[#2457B8]" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
            Target Variable Imbalance (Column: 'y')
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
          {/* Visual Percentage Bar */}
          <div className="md:col-span-7">
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs font-semibold text-[#172B4D] mb-1">
                  <span>Non-Subscribers ('no')</span>
                  <span>
                    {summary.target_distribution?.no?.count.toLocaleString()} (
                    {summary.target_distribution?.no?.percentage}%)
                  </span>
                </div>
                <div className="h-4 bg-[#F1F5F9] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#64748B] rounded-full"
                    style={{ width: `${summary.target_distribution?.no?.percentage}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold text-[#172B4D] mb-1">
                  <span className="text-[#2457B8]">Subscribers ('yes')</span>
                  <span className="text-[#2457B8] font-bold">
                    {summary.target_distribution?.yes?.count.toLocaleString()} (
                    {summary.target_distribution?.yes?.percentage}%)
                  </span>
                </div>
                <div className="h-4 bg-[#F1F5F9] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#2457B8] rounded-full"
                    style={{ width: `${summary.target_distribution?.yes?.percentage}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Context Note */}
          <div className="md:col-span-5 bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-4 text-xs leading-relaxed text-[#475569]">
            <p className="mb-2">
              <strong className="text-[#172B4D]">Imbalance Impact on Modeling: </strong>
              Because only ~11.7% of customers subscribe, naive models can trivially reach 88.3% accuracy by always predicting "no", while missing every single deposit.
            </p>
            <p>
              This is why the project uses <span className="font-semibold text-[#2457B8]">class_weight="balanced"</span> and measures success via F1-Score (0.4797) and ROC-AUC (0.8010).
            </p>
          </div>
        </div>
      </div>

      {/* 4. Feature Catalogue Table */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 mb-4 border-b border-[#EDF1F7]">
          <div className="flex items-center gap-2">
            <TableIcon className="w-4 h-4 text-[#2457B8]" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
              Feature Catalogue ({filteredFeatures.length} of {features.length} Features)
            </h2>
          </div>

          {/* Controls: Search + Filter */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-[#94A3B8]" />
              <input
                type="text"
                value={featureSearch}
                onChange={(e) => setFeatureSearch(e.target.value)}
                placeholder="Search features..."
                className="pl-8 pr-3 py-1 text-xs border border-[#CBD5E1] rounded-md outline-none focus:border-[#2457B8] w-40 sm:w-48 bg-white"
              />
            </div>

            <select
              value={featureFilter}
              onChange={(e) => setFeatureFilter(e.target.value)}
              className="text-xs border border-[#CBD5E1] rounded-md px-2 py-1 outline-none bg-white text-[#475569]"
            >
              <option value="all">All Features</option>
              <option value="numerical">Numerical Only</option>
              <option value="categorical">Categorical Only</option>
              <option value="inputs">Model Inputs Only</option>
              <option value="excluded">Excluded / Target</option>
            </select>
          </div>
        </div>

        {/* Features Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#E2E8F0] text-[#64748B] bg-[#F8FAFC]">
                <th className="py-2.5 px-3 font-semibold">Feature Name</th>
                <th className="py-2.5 px-3 font-semibold">Data Type</th>
                <th className="py-2.5 px-3 font-semibold">Role</th>
                <th className="py-2.5 px-3 font-semibold">Model Status</th>
                <th className="py-2.5 px-3 font-semibold">Description & Notes</th>
                <th className="py-2.5 px-3 font-semibold">Observed Values / Categories</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#F1F5F9]">
              {filteredFeatures.map((f) => (
                <tr
                  key={f.name}
                  className={
                    f.name === 'duration'
                      ? 'bg-[#FEF2F2] hover:bg-[#FEE2E2]'
                      : 'hover:bg-[#F8FAFC]'
                  }
                >
                  <td className="py-2.5 px-3 font-mono font-bold text-[#172B4D]">
                    {f.name}
                  </td>
                  <td className="py-2.5 px-3 text-[#475569]">{f.data_type}</td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                        f.role === 'Model Input'
                          ? 'bg-[#EBF3FF] text-[#2457B8]'
                          : f.role === 'Excluded Feature'
                          ? 'bg-[#FEE2E2] text-[#B91C1C]'
                          : f.role === 'Derived Feature'
                          ? 'bg-[#F0FDF4] text-[#15803D]'
                          : 'bg-[#F1F5F9] text-[#475569]'
                      }`}
                    >
                      {f.role}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    {f.used_in_model ? (
                      <span className="inline-flex items-center gap-1 text-[#15803D] font-medium">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Used
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[#B91C1C] font-semibold">
                        <XCircle className="w-3.5 h-3.5" /> Excluded
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 px-3 max-w-xs text-[#334155] leading-snug">
                    <div>{f.description}</div>
                    {f.notes && (
                      <div className="text-[10px] text-[#64748B] mt-0.5 italic">
                        {f.notes}
                      </div>
                    )}
                  </td>
                  <td className="py-2.5 px-3 max-w-xs text-[11px] text-[#64748B] font-mono truncate">
                    {f.examples}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. Read-Only Paginated Data Preview */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 mb-4 border-b border-[#EDF1F7]">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-[#2457B8]" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
              Read-Only Data Preview (10 records per page)
            </h2>
          </div>

          {/* Pagination Controls */}
          {previewData && (
            <div className="flex items-center gap-2 self-end sm:self-auto">
              <span className="text-xs text-[#64748B]">
                Page {previewData.current_page} of {previewData.total_pages}
              </span>
              <button
                type="button"
                onClick={() => setPreviewPage((p) => Math.max(1, p - 1))}
                disabled={previewPage === 1 || isLoadingPreview}
                className="p-1 rounded border border-[#E2E8F0] hover:bg-[#F8FAFC] disabled:opacity-40 disabled:cursor-not-allowed"
                aria-label="Previous page"
              >
                <ChevronLeft className="w-4 h-4 text-[#475569]" />
              </button>
              <button
                type="button"
                onClick={() => setPreviewPage((p) => p + 1)}
                disabled={previewPage >= (previewData.total_pages || 1) || isLoadingPreview}
                className="p-1 rounded border border-[#E2E8F0] hover:bg-[#F8FAFC] disabled:opacity-40 disabled:cursor-not-allowed"
                aria-label="Next page"
              >
                <ChevronRight className="w-4 h-4 text-[#475569]" />
              </button>
            </div>
          )}
        </div>

        {/* Data Table */}
        <div className="overflow-x-auto border border-[#E2E8F0] rounded-lg">
          {isLoadingPreview ? (
            <div className="p-8 text-center text-xs text-[#64748B]">
              <div className="w-5 h-5 border-2 border-[#2457B8] border-t-transparent rounded-full animate-spin mx-auto mb-2" />
              Loading records...
            </div>
          ) : previewData && previewData.rows ? (
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="bg-[#F8FAFC] text-[#64748B] border-b border-[#E2E8F0]">
                  {previewData.columns.map((col) => (
                    <th key={col} className="py-2 px-2.5 font-semibold text-[11px] whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F1F5F9] font-mono text-[11px]">
                {previewData.rows.map((row, idx) => (
                  <tr key={idx} className="hover:bg-[#F8FAFC]">
                    {previewData.columns.map((col) => (
                      <td
                        key={col}
                        className={`py-2 px-2.5 whitespace-nowrap ${
                          col === 'y'
                            ? row[col] === 'yes'
                              ? 'text-[#15803D] font-bold bg-[#DCFCE7]/40'
                              : 'text-[#64748B]'
                            : col === 'duration'
                            ? 'text-[#B91C1C]'
                            : 'text-[#1E293B]'
                        }`}
                      >
                        {String(row[col])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      </div>

      {/* 6. Exploratory Data Analysis (EDA) Highlights */}
      {edaData && (
        <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
          <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
            <BarChart2 className="w-4 h-4 text-[#2457B8]" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
              Verified EDA Conversion Rates by Category
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            {/* By Previous Outcome */}
            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3.5">
              <span className="font-bold text-[#172B4D] block mb-2 text-xs">
                Previous Outcome (poutcome)
              </span>
              <div className="space-y-1.5">
                {edaData.subscription_rate_by_poutcome?.map((p) => (
                  <div key={p.category}>
                    <div className="flex justify-between text-[11px] text-[#475569] mb-0.5">
                      <span className="capitalize">{p.category}</span>
                      <span className="font-mono font-bold text-[#172B4D]">
                        {p.conversion_rate_pct}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-[#E2E8F0] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#2457B8] rounded-full"
                        style={{ width: `${Math.min(p.conversion_rate_pct, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* By Contact Method */}
            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3.5">
              <span className="font-bold text-[#172B4D] block mb-2 text-xs">
                Contact Channel (contact)
              </span>
              <div className="space-y-1.5">
                {edaData.subscription_rate_by_contact?.map((c) => (
                  <div key={c.category}>
                    <div className="flex justify-between text-[11px] text-[#475569] mb-0.5">
                      <span className="capitalize">{c.category}</span>
                      <span className="font-mono font-bold text-[#172B4D]">
                        {c.conversion_rate_pct}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-[#E2E8F0] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#2457B8] rounded-full"
                        style={{ width: `${Math.min(c.conversion_rate_pct * 3, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* By Housing Loan */}
            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3.5">
              <span className="font-bold text-[#172B4D] block mb-2 text-xs">
                Housing Loan (housing)
              </span>
              <div className="space-y-1.5">
                {edaData.subscription_rate_by_housing?.map((h) => (
                  <div key={h.category}>
                    <div className="flex justify-between text-[11px] text-[#475569] mb-0.5">
                      <span>{h.category}</span>
                      <span className="font-mono font-bold text-[#172B4D]">
                        {h.conversion_rate_pct}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-[#E2E8F0] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#2457B8] rounded-full"
                        style={{ width: `${Math.min(h.conversion_rate_pct * 3, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* By Top Jobs */}
            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3.5">
              <span className="font-bold text-[#172B4D] block mb-2 text-xs">
                Top Occupations (job)
              </span>
              <div className="space-y-1.5">
                {edaData.subscription_rate_by_job?.slice(0, 4).map((j) => (
                  <div key={j.category}>
                    <div className="flex justify-between text-[11px] text-[#475569] mb-0.5">
                      <span className="capitalize">{j.category}</span>
                      <span className="font-mono font-bold text-[#172B4D]">
                        {j.conversion_rate_pct}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-[#E2E8F0] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#2457B8] rounded-full"
                        style={{ width: `${Math.min(j.conversion_rate_pct * 3, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-3.5 pt-3 border-t border-[#EDF1F7] text-[11px] text-[#64748B] flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5 text-[#2457B8] shrink-0" />
            <span>
              <strong>Correlation vs. Causation Notice: </strong> These rates reflect historical conversion frequencies in the Portuguese banking campaign and indicate predictive associations, not causal outcomes.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
