# DEBUG_STAGE1 Logging

## Overview
Temporary debug logging added to Stage 1 fact extraction to diagnose why only links appear in Stage 2.

## How to Enable

Set environment variable:
```bash
export DEBUG_STAGE1=true
```

Or in Streamlit Cloud secrets:
```
DEBUG_STAGE1 = "true"
```

## What Gets Logged

### 1. Raw LLM Response
**File:** `debug_logs/debug_stage1_raw_{run_id}.json`

Contains:
- `run_id`: Timestamp-based unique ID
- `timestamp`: ISO timestamp
- `raw_candidate_facts`: Complete list of facts returned by LLM (before validation)
- `extraction_warnings`: Any warnings from LLM
- `total_raw_facts`: Count

**Console:** Prints each raw fact with category and confidence

### 2. Rejected Facts
**File:** `debug_logs/debug_stage1_rejected_{run_id}.json`

Contains:
- `run_id`: Same run ID
- `timestamp`: ISO timestamp
- `rejected_facts`: Array of rejected facts with:
  - `fact`: The fact text
  - `category`: Category assigned by LLM
  - `confidence`: Confidence score
  - `evidence`: Evidence quote
  - `rejection_reasons`: Array of strings explaining why rejected
- `total_rejected`: Count

**Rejection Reasons:**
- `"is_complete_fact check failed"` - Fact doesn't meet completeness criteria (< 8 words, fragment, etc.)
- `"confidence too low: X < 0.50"` - Confidence below threshold
- `"evidence quote not found in source text"` - Evidence substring not in source
- `"duplicate fact (exact match)"` - Already seen (case-insensitive)

**Console:** Prints each rejected fact with reasons

### 3. Accepted Facts
**File:** `debug_logs/debug_stage1_accepted_{run_id}.json`

Contains:
- `run_id`: Same run ID
- `timestamp`: ISO timestamp
- `accepted_facts`: Array of facts that passed validation:
  - `fact`: The fact text
  - `category`: Category
  - `confidence`: Confidence score
  - `evidence`: Evidence quote
- `total_accepted`: Count

**Console:** Prints each accepted fact

## File Locations

**Local Development:**
```
llm-behavior-job-outreach/debug_logs/
```

**Streamlit Cloud:**
Files will be written to the app's working directory. Check Streamlit Cloud logs for exact paths.

## Interpreting Results

### Scenario A: LLM extracts facts, validator rejects them
- `debug_stage1_raw.json` will have many facts
- `debug_stage1_rejected.json` will show why they were rejected
- `debug_stage1_accepted.json` will be empty or have few facts

**Action:** Fix validator logic (confidence threshold, evidence matching, completeness check)

### Scenario B: LLM doesn't extract facts
- `debug_stage1_raw.json` will have few/no facts (only links)
- `debug_stage1_rejected.json` will be empty or minimal
- `debug_stage1_accepted.json` will match raw

**Action:** Fix extraction prompt (too strict, unclear instructions, output format issues)

## Disabling

Set environment variable:
```bash
export DEBUG_STAGE1=false
```

Or remove/unset the variable.

## Cleanup

Debug logs are gitignored. To remove:
```bash
rm -rf debug_logs/
```

