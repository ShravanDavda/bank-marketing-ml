import React, { useState, useEffect } from 'react';
import {
  GitBranch,
  CheckCircle2,
  Code2,
  Layers,
  Cpu,
  ShieldCheck,
  AlertTriangle,
  Info,
  ArrowLeft,
  ArrowRight,
  Database,
  Wrench,
  BarChart3,
  Rocket,
} from 'lucide-react';
import { fetchPipelineData } from '../services/predictionApi';

// 5 Parent Phases grouping the 12 verified lifecycle stages
const PHASES = [
  {
    id: 'phase-1',
    number: 1,
    title: 'Data Understanding',
    stageNumbers: [1, 2, 3],
    badge: 'Stages 1–3',
    icon: Database,
    description: 'Raw dataset ingestion, schema audit, quality validation, and exploratory feature distributions.',
  },
  {
    id: 'phase-2',
    number: 2,
    title: 'Data Preparation',
    stageNumbers: [4, 5, 6],
    badge: 'Stages 4–6',
    icon: Wrench,
    description: 'Domain feature engineering, stratified 80/20 train/test split, and scikit-learn ColumnTransformer.',
  },
  {
    id: 'phase-3',
    number: 3,
    title: 'Model Development',
    stageNumbers: [7, 8, 9],
    badge: 'Stages 7–9',
    icon: Cpu,
    description: 'Baseline pipeline training across 4 algorithm families, comparison, and 5-fold CV hyperparameter tuning.',
  },
  {
    id: 'phase-4',
    number: 4,
    title: 'Final Evaluation & Interpretation',
    stageNumbers: [10, 11],
    badge: 'Stages 10–11',
    icon: BarChart3,
    description: 'Held-out test set evaluation (9,043 samples), confusion matrix analysis, and Gini feature importances.',
  },
  {
    id: 'phase-5',
    number: 5,
    title: 'Deployment',
    stageNumbers: [12],
    badge: 'Stage 12',
    icon: Rocket,
    description: 'Fitted pipeline serialization to joblib (35.4 MB) and real-time FastAPI inference serving.',
  },
];

export default function PipelinePage() {
  const [pipelineData, setPipelineData] = useState(null);
  const [activePhaseIndex, setActivePhaseIndex] = useState(0);
  const [selectedStageNumber, setSelectedStageNumber] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      const res = await fetchPipelineData();
      if (res.success) {
        setPipelineData(res.data);
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
        <p className="text-sm text-[#475569] font-medium">Loading ML Pipeline Workflow...</p>
      </div>
    );
  }

  if (error || !pipelineData) {
    return (
      <div className="bg-[#FEF2F2] border border-[#FECACA] rounded-xl p-6 text-center shadow-sm">
        <p className="text-sm font-semibold text-[#B91C1C] mb-1">Failed to load Pipeline data</p>
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

  const stages = pipelineData.stages || [];
  const modelSpecs = pipelineData.model_specifications || {};

  // Find active stage object
  const activeStage = stages.find((s) => s.number === selectedStageNumber) || stages[0];

  // Helper to switch stage and sync parent phase
  const navigateToStage = (stageNum) => {
    if (stageNum < 1 || stageNum > 12) return;
    setSelectedStageNumber(stageNum);
    const parentPhaseIdx = PHASES.findIndex((p) => p.stageNumbers.includes(stageNum));
    if (parentPhaseIdx !== -1 && parentPhaseIdx !== activePhaseIndex) {
      setActivePhaseIndex(parentPhaseIdx);
    }
  };

  // Helper when clicking parent phase card
  const handlePhaseClick = (phaseIdx) => {
    setActivePhaseIndex(phaseIdx);
    const targetPhase = PHASES[phaseIdx];
    // If current selected stage is not in this phase, default to first stage of this phase
    if (!targetPhase.stageNumbers.includes(selectedStageNumber)) {
      setSelectedStageNumber(targetPhase.stageNumbers[0]);
    }
  };

  const currentPhase = PHASES[activePhaseIndex] || PHASES[0];
  const currentPhaseStages = stages.filter((s) => currentPhase.stageNumbers.includes(s.number));

  return (
    <div className="space-y-6">
      {/* 1. Page Header */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <span className="text-[11px] font-bold uppercase tracking-wider bg-[#EBF3FF] text-[#2457B8] px-2.5 py-0.5 rounded-full border border-[#D0E2FF]">
                5-Phase Lifecycle
              </span>
              <span className="text-[11px] font-bold uppercase tracking-wider bg-[#F1F5F9] text-[#475569] px-2.5 py-0.5 rounded-full border border-[#E2E8F0]">
                12 Detailed Stages
              </span>
              <span className="text-[11px] text-[#64748B]">
                All stages verified against repository artifacts
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-[#172B4D] tracking-tight m-0">
              {pipelineData.title}
            </h1>
            <p className="text-xs sm:text-sm text-[#5E6C84] mt-1 max-w-3xl leading-relaxed">
              Explore how the Bank Marketing dataset moves through five structured phases: from raw CSV ingestion and domain feature engineering to hyperparameter tuning and deployed FastAPI inference.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-center shrink-0">
            <div className="bg-[#F8FAFC] border border-[#E2E8F0] px-3.5 py-2 rounded-lg text-xs font-semibold text-[#172B4D]">
              <span className="text-[#64748B] block text-[10px] uppercase font-bold tracking-wider">
                Production Artifact
              </span>
              <span>Random Forest (Tuned Pipeline)</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Parent Phases Navigation (5 Parent Tabs) */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-4 sm:p-5 shadow-sm">
        <div className="flex items-center justify-between mb-3.5">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#2457B8]" />
            <h2 className="text-xs sm:text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
              ML Lifecycle Phases
            </h2>
          </div>
          <span className="text-[11px] text-[#64748B]">
            Select a phase to view its component stages
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {PHASES.map((phase, idx) => {
            const isActive = idx === activePhaseIndex;
            const PhaseIcon = phase.icon;
            return (
              <button
                key={phase.id}
                onClick={() => handlePhaseClick(idx)}
                className={`text-left p-3.5 rounded-xl border transition-all cursor-pointer flex flex-col justify-between ${
                  isActive
                    ? 'bg-[#EBF3FF] border-[#2457B8] shadow-sm ring-1 ring-[#2457B8]'
                    : 'bg-[#F8FAFC] border-[#E2E8F0] hover:bg-white hover:border-[#CBD5E1]'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                        isActive
                          ? 'bg-[#2457B8] text-white'
                          : 'bg-[#E2E8F0] text-[#475569]'
                      }`}
                    >
                      Phase {phase.number}
                    </span>
                    <PhaseIcon
                      className={`w-4 h-4 ${
                        isActive ? 'text-[#2457B8]' : 'text-[#94A3B8]'
                      }`}
                    />
                  </div>
                  <div
                    className={`text-xs font-bold leading-tight mb-1 ${
                      isActive ? 'text-[#2457B8]' : 'text-[#172B4D]'
                    }`}
                  >
                    {phase.title}
                  </div>
                  <p className="text-[11px] text-[#64748B] line-clamp-2 leading-relaxed">
                    {phase.description}
                  </p>
                </div>

                <div className="mt-3 pt-2 border-t border-[#EDF1F7] flex items-center justify-between text-[10px]">
                  <span className="font-semibold text-[#64748B]">{phase.badge}</span>
                  <span
                    className={`font-bold ${
                      isActive ? 'text-[#2457B8]' : 'text-[#94A3B8]'
                    }`}
                  >
                    {phase.stageNumbers.length} stages
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* 3. Subtabs for Active Phase Stages */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-4 sm:p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 mb-3.5 border-b border-[#EDF1F7]">
          <div>
            <div className="flex items-center gap-2">
              <GitBranch className="w-4 h-4 text-[#2457B8]" />
              <h3 className="text-xs sm:text-sm font-bold text-[#172B4D] m-0">
                Phase {currentPhase.number}: {currentPhase.title} — Detailed Stages
              </h3>
            </div>
            <p className="text-[11px] text-[#64748B] mt-0.5">
              Click a stage subtab to inspect implementation details, code snippets, and verified findings.
            </p>
          </div>
          <span className="text-[11px] font-medium text-[#475569] bg-[#F1F5F9] px-2.5 py-1 rounded-md border border-[#E2E8F0] self-start sm:self-auto">
            Viewing Stage {selectedStageNumber} of 12
          </span>
        </div>

        {/* Stage Subtabs Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
          {currentPhaseStages.map((stage) => {
            const isSelected = stage.number === selectedStageNumber;
            return (
              <button
                key={stage.id}
                onClick={() => setSelectedStageNumber(stage.number)}
                className={`text-left p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between gap-3 ${
                  isSelected
                    ? 'bg-[#EBF3FF] border-[#2457B8] shadow-xs'
                    : 'bg-[#F8FAFC] border-[#E2E8F0] hover:bg-white hover:border-[#CBD5E1]'
                }`}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <span
                    className={`w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-bold shrink-0 ${
                      isSelected
                        ? 'bg-[#2457B8] text-white'
                        : 'bg-[#E2E8F0] text-[#475569]'
                    }`}
                  >
                    {stage.number}
                  </span>
                  <div className="truncate">
                    <span
                      className={`text-xs font-semibold block truncate ${
                        isSelected ? 'text-[#2457B8]' : 'text-[#172B4D]'
                      }`}
                    >
                      {stage.title}
                    </span>
                    <span className="text-[10px] text-[#64748B] block truncate">
                      {stage.short_desc}
                    </span>
                  </div>
                </div>

                <CheckCircle2
                  className={`w-4 h-4 shrink-0 ${
                    isSelected ? 'text-[#2457B8]' : 'text-[#94A3B8]'
                  }`}
                />
              </button>
            );
          })}
        </div>
      </div>

      {/* 4. Selected Stage Deep-Dive Card */}
      {activeStage && (
        <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
          {/* Header of Stage */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 mb-5 border-b border-[#EDF1F7]">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-[#2457B8] text-white flex items-center justify-center font-bold text-sm shrink-0">
                {activeStage.number}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-[#64748B]">
                    Phase {currentPhase.number} · Stage {activeStage.number} of 12
                  </span>
                </div>
                <h2 className="text-base sm:text-lg font-bold text-[#172B4D] tracking-tight m-0">
                  {activeStage.title}
                </h2>
                <p className="text-xs text-[#5E6C84] mt-0.5">{activeStage.short_desc}</p>
              </div>
            </div>

            <div className="flex items-center gap-2 self-start sm:self-auto">
              <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full bg-[#DCFCE7] text-[#15803D] border border-[#BBF7D0]">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Verified in Project</span>
              </span>
            </div>
          </div>

          {/* Core Information Grid (2x2) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
            {/* What it Means & Why Used */}
            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-4 space-y-3">
              <div>
                <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-[#172B4D] mb-1">
                  <Info className="w-3.5 h-3.5 text-[#2457B8]" />
                  <span>What It Means</span>
                </div>
                <p className="text-xs text-[#475569] leading-relaxed">
                  {activeStage.what_it_means}
                </p>
              </div>

              <div className="pt-2 border-t border-[#EDF1F7]">
                <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-[#172B4D] mb-1">
                  <Layers className="w-3.5 h-3.5 text-[#2457B8]" />
                  <span>Why Used in This Project</span>
                </div>
                <p className="text-xs text-[#475569] leading-relaxed">
                  {activeStage.why_used}
                </p>
              </div>
            </div>

            {/* What Was Done & Limitations */}
            <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-4 space-y-3">
              <div>
                <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-[#172B4D] mb-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#16A34A]" />
                  <span>Implementation Actions</span>
                </div>
                <p className="text-xs text-[#475569] leading-relaxed">
                  {activeStage.what_was_done}
                </p>
              </div>

              <div className="pt-2 border-t border-[#EDF1F7]">
                <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-[#172B4D] mb-1">
                  <AlertTriangle className="w-3.5 h-3.5 text-[#D97706]" />
                  <span>Limitations & Constraints</span>
                </div>
                <p className="text-xs text-[#475569] leading-relaxed">
                  {activeStage.limitations}
                </p>
              </div>
            </div>
          </div>

          {/* Inputs, Outputs & Code Snippet */}
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-[#FFFFFF] border border-[#E1E7F0] rounded-lg p-3.5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#64748B] block mb-1">
                  Input Schema / Data
                </span>
                <p className="text-xs font-mono text-[#172B4D] bg-[#F1F5F9] px-2.5 py-1.5 rounded-md">
                  {activeStage.inputs}
                </p>
              </div>

              <div className="bg-[#FFFFFF] border border-[#E1E7F0] rounded-lg p-3.5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#64748B] block mb-1">
                  Output Artifact / Result
                </span>
                <p className="text-xs font-mono text-[#172B4D] bg-[#F1F5F9] px-2.5 py-1.5 rounded-md">
                  {activeStage.outputs}
                </p>
              </div>
            </div>

            {/* Code Snippet */}
            {activeStage.code_snippet && (
              <div className="bg-[#0F172A] text-[#F8FAFC] rounded-lg p-4 font-mono text-xs overflow-x-auto">
                <div className="flex items-center justify-between text-[#94A3B8] text-[11px] pb-2 mb-2 border-b border-[#334155]">
                  <span className="flex items-center gap-1.5">
                    <Code2 className="w-3.5 h-3.5" />
                    <span>Project Source Code Verification</span>
                  </span>
                  <span>Python · scikit-learn</span>
                </div>
                <pre className="text-xs leading-relaxed text-[#E2E8F0]">
                  {activeStage.code_snippet}
                </pre>
              </div>
            )}

            {/* Key Findings Callout */}
            <div className="bg-[#F0FDF4] border border-[#BBF7D0] rounded-lg p-3.5 flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-[#16A34A] shrink-0 mt-0.5" />
              <div>
                <span className="text-xs font-bold text-[#15803D] block mb-0.5">
                  Verified Finding:
                </span>
                <p className="text-xs text-[#166534] leading-relaxed">
                  {activeStage.key_findings}
                </p>
              </div>
            </div>

            {/* Previous / Next Stage Sequential Navigation Buttons */}
            <div className="pt-4 mt-2 border-t border-[#EDF1F7] flex items-center justify-between">
              <button
                disabled={selectedStageNumber <= 1}
                onClick={() => navigateToStage(selectedStageNumber - 1)}
                className={`inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-md border transition-all ${
                  selectedStageNumber <= 1
                    ? 'opacity-40 cursor-not-allowed bg-[#F8FAFC] text-[#94A3B8] border-[#E2E8F0]'
                    : 'cursor-pointer bg-white text-[#172B4D] border-[#CBD5E1] hover:bg-[#F1F5F9]'
                }`}
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                <span>Previous Stage</span>
              </button>

              <span className="text-[11px] text-[#64748B]">
                Stage {selectedStageNumber} of 12
              </span>

              <button
                disabled={selectedStageNumber >= 12}
                onClick={() => navigateToStage(selectedStageNumber + 1)}
                className={`inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-md border transition-all ${
                  selectedStageNumber >= 12
                    ? 'opacity-40 cursor-not-allowed bg-[#F8FAFC] text-[#94A3B8] border-[#E2E8F0]'
                    : 'cursor-pointer bg-[#2457B8] text-white border-[#2457B8] hover:bg-[#1B4393]'
                }`}
              >
                <span>Next Stage</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 5. Shared Model Architecture & Deployment Specifications Card */}
      <div className="bg-white border border-[#E1E7F0] rounded-xl p-5 sm:p-6 shadow-sm">
        <div className="flex items-center gap-2 pb-3 mb-4 border-b border-[#EDF1F7]">
          <Cpu className="w-4 h-4 text-[#2457B8]" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-[#172B4D] m-0">
            Deployed Model Architecture Specifications
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-4">
            <span className="text-[#64748B] block mb-1 font-medium">Pipeline Architecture</span>
            <p className="font-bold text-sm text-[#172B4D] mb-1">{modelSpecs.model_name}</p>
            <p className="text-[#64748B] text-[11px] leading-relaxed">
              Serialized end-to-end via joblib into a single fitted artifact ({modelSpecs.artifact_size_mb} MB).
            </p>
          </div>

          <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-4">
            <span className="text-[#64748B] block mb-1 font-medium">Verified Hyperparameters</span>
            {modelSpecs.hyperparameters ? (
              <ul className="space-y-1 font-mono text-[11px] text-[#172B4D]">
                {Object.entries(modelSpecs.hyperparameters).map(([key, val]) => (
                  <li key={key}>
                    {key}: {val === null ? 'None' : typeof val === 'string' ? `'${val}'` : String(val)}
                  </li>
                ))}
              </ul>
            ) : (
              <ul className="space-y-1 font-mono text-[11px] text-[#172B4D]">
                <li>n_estimators: 100</li>
                <li>max_depth: None</li>
                <li>min_samples_leaf: 3</li>
                <li>class_weight: 'balanced'</li>
                <li>random_state: 42</li>
              </ul>
            )}
          </div>

          <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-4">
            <span className="text-[#64748B] block mb-1 font-medium">Preprocessing Specifications</span>
            <p className="text-[#172B4D] font-semibold mb-1">
              {modelSpecs.preprocessing_pipeline?.preprocessor_type}
            </p>
            <p className="text-[11px] text-[#64748B] leading-relaxed">
              StandardScaler on 7 numerical features; OneHotEncoder on 9 categorical features. Expands 16 inputs to 51 transformed features.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

