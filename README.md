# VIP Individual Project Part 2 — Beam Design and AI Model Comparison

This project compares SciPy SLSQP optimization with two AI design agents—Ollama Llama 3.2 1B and OpenAI GPT-4.1-mini—for minimizing the mass of a rectangular steel beam while satisfying deflection and stress constraints.

**Repository:** https://github.com/sais03687/VIP-Beam-AI-Model-Comparison

## Design problem

The unmodified instructor-supplied `beam_model.py` evaluates each design using dimensions in **millimeters**.

| Parameter | Value |
|---|---|
| Width bounds | 20–100 mm |
| Height bounds | 40–120 mm |
| Maximum midspan deflection | 1.0 mm |
| Maximum bending-surface von Mises stress | 120 MPa |
| Objective | Minimize beam mass (kg), subject to both limits |
| Shared starting design | 50 × 80 mm (31.4 kg) |

## Completed experiment results

| Method | Run | Best feasible mass | Best width × height | Evaluator calls | Invalid AI replies | Elapsed time |
|---|---:|---:|---|---:|---:|---:|
| SciPy SLSQP | 1 | 13.4233 kg | 20.000 × 85.499 mm | 152 | N/A | ~0.01 s |
| Ollama Llama 3.2 1B | 1 | 31.4000 kg | 50 × 80 mm | 12 | 0 | 45.07 s |
| Ollama Llama 3.2 1B | 2 | 31.4000 kg | 50 × 80 mm | 12 | 0 | 29.32 s |
| OpenAI GPT-4.1-mini | 1 | 13.4235 kg | 20 × 85.5 mm | 12 | 0 | 10.64 s |
| OpenAI GPT-4.1-mini | 2 | 13.4235 kg | 20 × 85.5 mm | 12 | 0 | 9.09 s |

Both AI models ran twice. Each run included the same starting design followed by **11 proposed designs**, for **44 AI proposals** in total. All 44 proposals were valid; none required repair or substitution. GPT reached a feasible design close to the SciPy optimum in both runs. Llama did not improve on the common starting design within its proposal budget.

The 152 SciPy calls include objective and constraint evaluations and floating-point feasibility checks. These checks are not invalid AI responses. The comparison uses unequal evaluation budgets and is not a controlled comparison of equal numbers of evaluations.

## Files

| File | Description |
|---|---|
| `beam_model.py` | Original instructor-provided beam evaluator; kept unchanged |
| `surams_beam_ai.py` | SciPy optimizer, Llama and GPT agents, validation, CSV logging, graph, and report generation |
| `surams_beam_ai.ipynb` | Notebook demonstrating the evaluator/optimizer and containing the full-run configuration and results viewer |
| `surams_beam_ai.csv` | Recorded SciPy calls and all four live AI runs |
| `surams_beam_ai.pdf` | Results report with method comparison and search-history graph |
| `requirements.txt` | Python dependencies |
| `surams_beam_ai_search_history.png` | Optional standalone search-history plot; also embedded in PDF |
| `run_metadata.json` | Optional optimizer diagnostics and model settings |

## Installation and prerequisites

Requires Python 3.10 or newer, Ollama, and access to the OpenAI API.

From the repository folder, run:

```powershell
python -m pip install -r requirements.txt
ollama pull llama3.2:1b
```

Ensure Ollama is running locally. If necessary, start it with `ollama serve` in a separate terminal. Set your own OpenAI API key in the PowerShell session before a live GPT run:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

Never put an actual API key in source files, notebook outputs, commits, or the README.

## Running the experiment

Run all methods and regenerate the CSV, graph, report, and metadata:

```powershell
python surams_beam_ai.py --github-url "https://github.com/sais03687/VIP-Beam-AI-Model-Comparison"
```

**Warning:** This command runs fresh model trials and **overwrites the existing recorded outputs**. The committed CSV contains the completed experimental results; you do not need to rerun the models to review them.

To run only the numerical optimizer:

```powershell
python surams_beam_ai.py --optimizer-only
```

This also regenerates output files for the optimizer-only run, so use a separate output directory if you want to preserve the completed results:

```powershell
python surams_beam_ai.py --optimizer-only --output-dir scipy_test
```

To inspect the saved results without making model calls, open `surams_beam_ai.csv` or use the results-viewing cell in `surams_beam_ai.ipynb`.

## Experimental procedure

- Both models use the same prompt-building function, proposal validator, starting design (50 × 80 mm), and sequential evaluation feedback.
- The models are `llama3.2:1b` (local Ollama) and `gpt-4.1-mini` (OpenAI API), with temperature set to `0.2`.
- Every AI response must be a JSON object containing exactly numeric `width_mm` and `height_mm` within the allowed bounds. Invalid responses would consume a proposal opportunity without calling the beam evaluator.
- A candidate becomes the current best only if it satisfies **both** physical limits and has lower mass than the previous best feasible design.
- The CSV logs evaluated designs, feasibility, response statuses, elapsed time, and best feasible mass throughout each run.

## Limitations

The shared black-box evaluator uses linear-elastic, small-deflection beam theory rather than finite-element analysis. It excludes self-weight, local contact effects, impacts, and other real-world complexities. Results may vary with model versions, prompting, sampling settings, and API behavior. Only two trials per AI model were conducted, so these findings should not be treated as a broad performance benchmark.
