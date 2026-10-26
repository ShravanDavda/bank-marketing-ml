import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  Landmark,
  Calculator,
  GitBranch,
  BarChart3,
  Database,
  Menu,
  X,
  Activity,
} from 'lucide-react';

export default function Navigation({ isConnected = true, checkingHealth = false }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    {
      to: '/predict',
      label: 'Predict Subscription',
      icon: Calculator,
      description: 'Single-customer scoring',
    },
    {
      to: '/pipeline',
      label: 'ML Pipeline',
      icon: GitBranch,
      description: '12-stage workflow',
    },
    {
      to: '/performance',
      label: 'Model Performance',
      icon: BarChart3,
      description: 'Test metrics & evaluation',
    },
    {
      to: '/dataset',
      label: 'Dataset Explorer',
      icon: Database,
      description: 'Dataset statistics & EDA',
    },
  ];

  return (
    <header className="bg-white border border-[#E1E7F0] rounded-xl px-4 sm:px-6 py-3.5 mb-6 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#EBF3FF] text-[#2457B8] flex items-center justify-center shrink-0">
            <Landmark className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base sm:text-lg font-bold text-[#172B4D] tracking-tight">
                Bank Marketing Predictor
              </span>
              <span className="hidden lg:inline-flex text-[10px] font-semibold bg-[#EBF3FF] text-[#2457B8] px-2 py-0.5 rounded-full border border-[#D0E2FF]">
                ML Showcase
              </span>
            </div>
            <p className="text-[11px] text-[#5E6C84] hidden sm:block">
              Term deposit conversion intelligence & machine learning pipeline
            </p>
          </div>
        </div>

        {/* Desktop Navigation Links */}
        <nav className="hidden md:flex items-center gap-1 bg-[#F8FAFC] p-1 rounded-lg border border-[#E2E8F0]">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                    isActive
                      ? 'bg-white text-[#2457B8] shadow-xs border border-[#E1E7F0]'
                      : 'text-[#475569] hover:text-[#172B4D] hover:bg-[#F1F5F9]'
                  }`
                }
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Right Status Badges & Mobile Menu Button */}
        <div className="flex items-center gap-2.5">
          {/* API Health Status */}
          <div
            className="inline-flex items-center gap-1.5 bg-[#F8FAFC] border border-[#E2E8F0] px-2.5 py-1 rounded-md text-[11px] font-medium text-[#475569]"
            title={isConnected ? 'Backend API operational' : 'Backend API unreachable on port 8000'}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                checkingHealth
                  ? 'bg-amber-400 animate-pulse'
                  : isConnected
                  ? 'bg-[#36B37E]'
                  : 'bg-red-500'
              }`}
            />
            <span className="hidden xs:inline">
              {checkingHealth ? 'Connecting...' : isConnected ? 'API Online' : 'API Offline'}
            </span>
          </div>

          {/* Model Tag */}
          <div className="hidden xl:inline-flex items-center gap-1 bg-[#EBF3FF] text-[#2457B8] border border-[#D0E2FF] px-2.5 py-1 rounded-md text-[11px] font-semibold">
            <Activity className="w-3 h-3" />
            <span>Random Forest (Tuned)</span>
          </div>

          {/* Mobile Menu Hamburger */}
          <button
            type="button"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-1.5 rounded-lg border border-[#E2E8F0] text-[#475569] hover:text-[#172B4D] hover:bg-[#F8FAFC]"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Dropdown Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden pt-3 mt-3 border-t border-[#EDF1F7] flex flex-col gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={() => setMobileMenuOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                    isActive
                      ? 'bg-[#EBF3FF] text-[#2457B8]'
                      : 'text-[#475569] hover:bg-[#F8FAFC] hover:text-[#172B4D]'
                  }`
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                <div>
                  <div className="leading-tight">{item.label}</div>
                  <div className="text-[10px] text-[#64748B] font-normal">{item.description}</div>
                </div>
              </NavLink>
            );
          })}
        </div>
      )}
    </header>
  );
}
