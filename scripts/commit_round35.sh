#!/usr/bin/env bash
set -euo pipefail

cd /Users/ayoubalhari/Downloads/crypto

if [ -x .venv/bin/python ]; then
  source .venv/bin/activate
fi

python -m compileall -q \
  core data research reporting execution portfolio ml scripts tests

ruff check --select F,E9 \
  core data research reporting execution portfolio ml scripts tests

python -m pytest -q \
  tests/test_round34_active_portfolio.py \
  tests/test_round35_active_live_execution.py \
  tests/test_round33_whole_stack_integration.py

python -m pytest -q
python scripts/active_live_contract_audit.py
git diff --check

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "ABORT: .env is tracked. Remove it from Git before committing."
  exit 2
fi

git add -A

# Never commit secrets, environments, market data or generated runtime state.
for path in \
  .env \
  .venv \
  output \
  data_store \
  config/live_playbook_authority.json \
  config/live_strategy_approvals.yaml \
  config/inventory_risk_override.json; do
  git restore --staged -- "$path" 2>/dev/null || true
done

bad="$(
  git diff --cached --name-only |
  grep -E '(^|/)\.env($|\.)|(^|/)\.venv/|^output/|^data_store/|^config/live_playbook_authority\.json$|^config/live_strategy_approvals\.yaml$|^config/inventory_risk_override\.json$' || true
)"
if [ -n "$bad" ]; then
  echo "ABORT: unsafe staged paths:"
  echo "$bad"
  exit 3
fi

git diff --cached --check

if git diff --cached --quiet; then
  echo "Nothing staged; no commit created."
  exit 0
fi

echo "Staged source changes:"
git status --short

git commit -m "feat: activate portfolio-driven live execution"
git push origin "$(git branch --show-current)"
