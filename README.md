# LLM Behavior Job Outreach

A simple Python project that measures LLM reliability for job application cold outreach messages using **Groq API** (Llama3 model).

## What "LLM Behavior" Means Here

This tool evaluates how well AI models generate job outreach messages (emails and LinkedIn DMs) by measuring:

1. **Constraint Following**: Does it respect word limits, include required items, and maintain professional tone?
2. **Fact Accuracy**: Does it avoid fabricating information not in the allowed facts list?
3. **Stability**: Are results consistent across multiple runs?
4. **Self-Awareness**: Does it know when it's uncertain (not overconfident when failing checks)?

These behaviors matter because in real job applications, you need:
- Messages that follow your requirements exactly
- No made-up facts that could damage your credibility
- Consistent quality across multiple generations
- Honest confidence levels

## Project Structure

```
llm-behavior-job-outreach/
├── src/
│   ├── run.py          # Main evaluation script
│   ├── checks.py       # Validation functions
│   └── prompts.json    # Test prompts (8 job outreach examples)
├── results/            # Generated at runtime
│   ├── outputs.csv    # Detailed results per run
│   └── summary.json   # Aggregated metrics
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   Create a `.env` file in the project root:
   ```bash
   # Create .env file
   cat > .env << EOF
   GROQ_API_KEY=your_key_here
   MODEL=llama3-8b-8192
   RUNS_PER_PROMPT=3
   TEMPERATURE=0.2
   EOF
   ```
   Then edit `.env` and replace `your_key_here` with your actual Groq API key.
   
   **Get Groq API Key:** https://console.groq.com/keys

3. **Optional configuration** (in `.env`):
   - `MODEL`: Model to test (default: `llama3-8b-8192`)
   - `RUNS_PER_PROMPT`: Number of runs per prompt (default: `3`)
   - `TEMPERATURE`: Sampling temperature (default: `0.2`)

## How to Run

```bash
python src/run.py
```

The script will:
1. Load prompts from `src/prompts.json`
2. Generate 3 variants of each outreach message (configurable)
3. Check each message for:
   - Word limit compliance
   - Required items included
   - Professional tone
   - No unauthorized facts
4. Save results to `results/outputs.csv` and `results/summary.json`
5. Print a summary table to the terminal

## Understanding the Metrics

### Pass Rate
Percentage of runs where all checks pass (word limit + must-include + tone + no new facts).

**Good**: > 0.8 (80%+ of runs pass all checks)  
**Concerning**: < 0.6 (model frequently violates constraints)

### Stability Rate
Percentage of prompts where all 3 runs have the same pass/fail outcome.

**Good**: > 0.7 (model is consistent)  
**Concerning**: < 0.5 (model behavior varies unpredictably)

### Overconfidence Rate
Percentage of prompts where the model reports high confidence (≥0.75) but fails checks.

**Good**: < 0.2 (model knows when it's uncertain)  
**Concerning**: > 0.4 (model is overconfident when it shouldn't be)

## Test Prompts

The project includes 8 example prompts covering:
- **Channels**: Email and LinkedIn DM
- **Recipients**: Recruiters, Hiring Managers, Founders
- **Companies**: Google, Meta, Anthropic, Goldman Sachs, OpenAI, Cohere, McKinsey, Apple
- **Roles**: Software Engineer, ML Engineer, Research Scientist, Data Scientist, Quantitative Developer

## Adding New Prompts

Edit `src/prompts.json` and add a new entry:

```json
{
  "id": "your_prompt_id",
  "channel": "email" or "linkedin_dm",
  "recipient_type": "recruiter" or "hiring_manager" or "founder",
  "company": "Company Name",
  "target_role": "Job Title",
  "tone": "professional",
  "max_words": 150,
  "allowed_facts": ["fact1", "fact2", "fact3"],
  "must_include": ["GitHub", "NYU", "Ask for chat"],
  "notes": "Your rough notes or draft here"
}
```

Then run the evaluation again.

## Output Files

### outputs.csv
One row per run with columns:
- `id`: Prompt identifier
- `run_idx`: Run number (0, 1, 2)
- `channel`: Email or LinkedIn DM
- `company`: Target company
- `target_role`: Job role
- `confidence`: Model's self-reported confidence (0-1)
- `within_word_limit`: Word limit check result
- `must_include_ok`: Required items check result
- `adds_new_facts`: Whether new facts were detected
- `tone_ok`: Professional tone check result
- `overall_pass`: Whether all checks passed
- `message`: The generated message
- `notes`: Check notes/errors

### summary.json
Aggregated metrics:
- `overall`: Overall pass rate, stability, overconfidence
- `by_channel`: Breakdown for email vs LinkedIn DM
- `by_recipient_type`: Breakdown for recruiter vs hiring_manager vs founder
- `by_prompt`: Per-prompt metrics

## Why This Matters for LLM Reliability

Job outreach is a critical use case where LLM failures have real consequences:
- **Constraint violations** can make you look unprofessional
- **Fabricated facts** can damage your credibility
- **Inconsistent outputs** make it hard to choose the best message
- **Overconfidence** means the model doesn't know when it's making mistakes

This tool helps you measure these behaviors before using LLMs in production.

## Next Steps / Future Enhancements

1. **Semantic Similarity**: Compare messages across runs to measure consistency using embeddings
2. **RAG (Retrieval-Augmented Generation)**: Add fact-checking against a knowledge base
3. **Human Labels**: Collect human judgments on message quality and effectiveness
4. **A/B Templates**: Test different message templates and structures
5. **Response Rate Tracking**: Track actual response rates from generated messages
6. **Multi-Model Comparison**: Test multiple models side-by-side
7. **Cost Tracking**: Monitor API costs per evaluation

## Example Output

```
================================================================================
LLM BEHAVIOR EVALUATION - JOB OUTREACH
================================================================================

Overall Metrics:
  Pass Rate:         0.875
  Stability Rate:    0.750
  Overconfidence:    0.125

By Channel:
  Email:
    Pass Rate:         0.900
    Stability Rate:    0.800
    Overconfidence:    0.100
  Linkedin Dm:
    Pass Rate:         0.850
    Stability Rate:    0.700
    Overconfidence:    0.150

By Recipient Type:
  Recruiter:
    Pass Rate:         0.875
    Stability Rate:    0.750
    Overconfidence:    0.125
```

## License

This project is provided as-is for evaluation purposes.

