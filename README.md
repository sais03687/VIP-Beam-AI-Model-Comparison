# VIP Individual Project Part 2 — Beam design and AI model comparison

This project uses the student's initial `surams_beam_ai.py` evaluation and the objective/constraints from `beam_evaluation_VIP_fa26.py` and Lecture 6. The supplied `beam_model.py` is unchanged. `evaluate_beam` accepts **millimeters**; the assignment PDF has one conflicting sentence saying meters, but its sample call and actual code are in millimeters.

## Requirements

- Python 3.10 or newer
- `pip install -r requirements.txt`
- Ollama installed and running locally, with `ollama pull llama3.2:1b`
- An OpenAI API key, configured as environment variable `OPENAI_API_KEY` (never store it in the code or repository)
- Model `gpt-4.1-mini` available through the OpenAI API (can change with `--gpt-model`)

## Run

From this directory:

```powershell
python -m pip install -r requirements.txt
ollama pull llama3.2:1b
# Have Ollama running (ollama serve if necessary)
$env:OPENAI_API_KEY="YOUR_API_KEY"  # set locally only; do not paste in submitted files
python surams_beam_ai.py --github-url "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY"
```

Run just the SciPy baseline, without contacting model providers:

```powershell
python surams_beam_ai.py --optimizer-only
```

**Do not submit the pre-generated baseline-only CSV and PDF as final results.** Running the full command overwrites them with actual model evaluations. Each Llama and GPT agent gets two fresh runs with the 50 x 80 mm starting design, followed by exactly 11 proposal opportunities per run. The same prompt constructor and JSON validator are used for both models. Replies are never repaired, clipped, or substituted. Failure to reach the service is logged and consumes a proposal opportunity; repair the service and rerun both trials for valid comparisons.

## Outputs

- `surams_beam_ai.ipynb` — notebook to import and run the experiment
- `surams_beam_ai.py` — command-line experiment runner
- `surams_beam_ai.csv` — all evaluated calls, plus attempted AI proposals
- `surams_beam_ai.pdf` — report with table, convergence summary and AI histories
- `surams_beam_ai_search_history.png` — plot embedded in the report
- `run_metadata.json` — model settings and optimizer diagnostics

`surams_beam_ai.csv` records every SciPy evaluation (objective, constraint and final evaluations) as well as every attempted model proposal. Multiple SciPy rows may have identical candidates because the optimizer evaluates the same design through different function callbacks. `run_metadata.json` counts all black-box calls. Note the optimizer optimum lies on the deflection boundary; the runner logs extremely small numerical interior checks to ensure the final selected design meets the unchanged evaluator's *strict* feasibility flag.

## Reproducibility and honesty

The local environment used to prepare this package did not have an Ollama executable or an OpenAI API key. Only the numerical optimization was executed. AI results are **not** simulated or fabricated. Run the full experiment on your Windows machine to produce the final CSV, figure, and PDF. Add your actual GitHub repository URL and push `beam_model.py`, the notebook, main script, results CSV, and README. Do not commit secrets, `.env`, or environments.

## Limitations

The provided black box is a beam-theory calculation, not finite-element analysis. It excludes self-weight, contact concentration, large deflections and dynamic loads. Agent performance can be affected by temperature, language-model version and stochastic proposal sequences. The optimizer and agents share the evaluator but use very different evaluation budgets.
