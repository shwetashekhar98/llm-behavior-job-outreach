# High-Stakes Claim Verification Layer

## Feature Overview

This is an **additive, non-breaking feature** that adds trust calibration for high-stakes claims (e.g., NeurIPS publications, PhD degrees, awards) without changing default behavior.

## Feature Flags

### Location
Feature flags are defined at the top of `streamlit_app.py`:

```python
ENABLE_HIGH_STAKES_LAYER = False  # Default: OFF (non-breaking)
ENFORCE_HIGH_STAKES_LANGUAGE = False  # Default: OFF (non-breaking)
```

### How to Enable

1. **Enable High-Stakes Layer (UI only)**:
   - Set `ENABLE_HIGH_STAKES_LAYER = True` in `streamlit_app.py`
   - OR check the "Enable High-Stakes Claim Layer" checkbox in the UI (only visible if flag is True)

2. **Enable Language Enforcement (generation)**:
   - Set `ENFORCE_HIGH_STAKES_LANGUAGE = True` in `streamlit_app.py`
   - This requires `ENABLE_HIGH_STAKES_LAYER` to be effective

## What Gets Flagged as High-Stakes

A fact is considered high-stakes if:

1. **Category-based**: Category is one of:
   - `impact`
   - `awards`
   - `education`

2. **Keyword-based**: Fact text contains (case-insensitive) any of:
   - Conference names: `neurips`, `icml`, `cvpr`, `acl`, `emnlp`, `nips`
   - Companies: `openai`, `google`, `meta`, `amazon`, `microsoft`, `apple`
   - Universities: `harvard`, `mit`, `stanford`
   - Credentials: `phd`
   - Organizations: `ieee`, `acm`, `nasa`

## Stage 2 UI Behavior (when enabled)

When `ENABLE_HIGH_STAKES_LAYER` is ON:

1. **Warning Labels**: High-stakes facts show:
   ```
   ⚠️ High-Stakes Claim — verification recommended
   ```

2. **Verification Toggle**: Each high-stakes fact has:
   - Dropdown: "unverified" or "verified"
   - URL input field (required if "verified" is selected)
   - Warning if verified but URL is empty

3. **Summary Section**: Shows:
   - Total high-stakes facts
   - Count of verified facts
   - Count of unverified facts

## Data Model

Facts with high-stakes metadata include:
- `trust_flag`: "high_stakes" or "normal"
- `verification_status`: "verified" or "unverified"
- `verification_url`: URL string (empty if unverified)

**Important**: Existing fact structure is preserved. New fields are added, not replaced.

## Generation Behavior (when ENFORCE_HIGH_STAKES_LANGUAGE is ON)

If a high-stakes fact is approved but unverified, the generation prompt includes:
- Instruction to use cautious phrasing: "According to my profile, ..." or "As noted in my background, ..."
- Metadata about which facts are unverified

## Manual Test Steps

### Test 1: Default Behavior (Flags OFF)
1. Set both flags to `False`
2. Extract facts from a profile containing "Published a paper at NeurIPS 2025"
3. **Expected**: No high-stakes warnings, normal fact approval flow
4. **Result**: ✅ Behavior identical to before

### Test 2: High-Stakes Layer ON (UI only)
1. Set `ENABLE_HIGH_STAKES_LAYER = True`
2. Extract facts from a profile containing "Published a paper at NeurIPS 2025"
3. **Expected**: 
   - Fact shows "⚠️ High-Stakes Claim" warning
   - Verification dropdown appears
   - Summary shows high-stakes count
4. **Result**: ✅ UI shows warnings and verification options

### Test 3: Language Enforcement ON
1. Set both flags to `True`
2. Extract facts with NeurIPS claim
3. Mark as "unverified" (or leave default)
4. Generate message
5. **Expected**: Generation prompt includes cautious phrasing instruction
6. **Result**: ✅ Generation uses cautious language for unverified claims

## Files Modified

1. **`streamlit_app.py`**:
   - Added feature flags
   - Added high-stakes UI in Stage 2
   - Added checkbox for user control
   - Passes metadata to evaluation

2. **`src/high_stakes.py`** (NEW):
   - `is_high_stakes()` function
   - `annotate_fact_with_trust()` function

3. **`src/evaluation_runner.py`**:
   - Added optional parameters for high-stakes metadata
   - Added language enforcement in generation prompt (if enabled)

## Backward Compatibility

✅ **All changes are backward compatible**:
- Default behavior unchanged (flags OFF)
- Existing fact structure preserved
- No breaking changes to validation
- No changes to extraction prompt
- No changes to fabrication/unsupported validators
- No changes to must-include behavior

## Example

**Input fact**: "Published a research paper at NeurIPS 2025 on large language model reliability."

**With flags OFF**: Treated as normal fact, no special handling.

**With `ENABLE_HIGH_STAKES_LAYER = True`**:
- Shows warning: "⚠️ High-Stakes Claim — verification recommended"
- User can mark as verified/unverified
- If verified, requires URL (e.g., paper link)
- Summary shows: "High-Stakes Facts: 1, Verified: 0, Unverified: 1"

**With `ENFORCE_HIGH_STAKES_LANGUAGE = True`** (and unverified):
- Generation prompt includes: "For unverified high-stakes claims, use cautious phrasing..."
- Generated message might say: "According to my profile, I published a research paper at NeurIPS 2025..."

