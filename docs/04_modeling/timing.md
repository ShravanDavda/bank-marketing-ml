# Bank Marketing ML Project — Commit Schedule

**Project deadline:** October 15, 2026, 6:00 PM  
**Target completion:** October 15, 2026, 4:00 PM  
**Repository workflow:** Shared `main` branch  
**Rule:** Each member must verify their work, commit, and push before the next member starts.

| Date & Time | Member | Work to Complete and Commit |
|---|---|---|
| Oct 14 · 12:00 PM | Coordinator | Initial repository setup, README, `.gitignore`, dependencies, and dataset path verification |
| 12:15 PM | Member 1 | Section 1 — Problem definition |
| 12:30 PM | Member 1 | Section 2 — Imports |
| 12:45 PM | Member 1 | Section 3 — Dataset loading |
| 1:15 PM | Member 1 | Section 4 — Data understanding |
| 2:30 PM | Member 1 | Section 6 — EDA and observations |
| 3:00 PM | Member 1 | Final verification and commit |
| 3:15 PM | Member 2 | Section 5 — Data cleaning |
| 4:00 PM | Member 2 | Section 7 — Feature engineering |
| 4:30 PM | Member 2 | Section 8 — Train-test split |
| 5:30 PM | Member 2 | Section 9 — Preprocessing |
| 6:00 PM | Member 2 | Verify, commit, and push |
| Oct 15 · 9:00 AM | Member 3 | Section 10 — Baseline model training |
| 10:15 AM | Member 3 | Section 11 — Evaluation metrics |
| 11:15 AM | Member 3 | Section 12 — Model comparison |
| 11:45 AM | Member 3 | Verify, commit, and push |
| 12:15 PM | Member 4 | Sections 13.1–13.3 — CV, scorers, and grids |
| 12:45 PM | Member 4 | Section 13.4 — GridSearchCV |
| 2:00 PM | Member 4 | Section 13.5 — Compare tuned candidates |
| 2:30 PM | Member 4 | Verify, commit, and push |
| 2:45 PM | Member 5 | Section 14 — Final evaluation |
| 3:15 PM | Member 5 | Section 15 — Feature importance |
| 3:45 PM | Member 5 | Section 16 — Conclusion |
| 4:00 PM | Member 5 | Target final commit and push |
| 4:00–5:00 PM | All members | Restart kernel, run all notebook cells, and fix integration issues |
| 5:00–6:00 PM | Coordinator | Final QA, corrections, and final commit if needed |

## Git Commit Rules

1. Work in the existing notebook: `notebooks/bank_marketing_ml.ipynb`.
2. Follow the section order and do not delete another member's work.
3. Before editing, pull the latest changes from `main`.
4. Commit only completed, verified work.
5. Push changes immediately after committing.
6. Do not commit the local dataset `data/bank-full.csv`.
7. Avoid simultaneous notebook edits to prevent merge conflicts.
8. After all sections are complete, restart the kernel and execute the entire notebook from top to bottom.

## Final Completion Checklist

- [ ] All 16 sections are present and in the correct order.
- [ ] Dataset loads using the correct relative path.
- [ ] Preprocessing and model training run without errors.
- [ ] Cross-validation and hyperparameter tuning complete successfully.
- [ ] Model metrics and comparison tables display correctly.
- [ ] `duration` is excluded from predictive model features.
- [ ] Feature importance and conclusion match the actual results.
- [ ] Notebook runs from a fresh kernel without errors.
- [ ] All changes are committed and pushed to `main`.
- [ ] README and dependencies are up to date.

**Important:** The 4:00 PM target is the completion goal. Use 4:00–6:00 PM as the final testing and correction buffer. Do not sacrifice notebook correctness just to meet the target.