# Teacher Supervision, Fixed Representations and Calibration

Paper and simulation code for

> A. Sudjianto and D. Burakov, *Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models*, October 2026.

A candidate-scoring decision model, such as Jev, returns a probability for each answer in a set of candidates. Public documentation for Jev and JevAny describes three training ingredients:

- soft targets or reference probabilities from larger models
- a frozen backbone
- temperature calibration

The paper studies each ingredient in a Gaussian setting where the risks can be computed exactly. It makes no claim about how Jev itself works internally.

## Contents

| File | Description |
|---|---|
| `paper.tex`, `paper.pdf` | LaTeX source and compiled paper |
| `experiments.py` | Monte Carlo checks of every formula in the paper; writes the figure and `results.json` |
| `results.json` | Numerical output of the run reported in the paper |
| `validation_sweeps.pdf`, `validation_sweeps.png` | Figure 1 of the paper |

## Results

**Least squares with Gaussian random design**

- Exact finite-sample risk for three ways of training:
  - on teacher targets
  - on a fixed linear compression of the predictors
  - on a mixture of teacher targets and outcomes, where the teacher may use directions that the compression discards and the two target errors may be correlated (Theorem 4.3)
- A closed-form optimal teacher weight.
- A projection identity for the teacher bias that a compressed student inherits.

**Gaussian-probit outcome**

- The excess Brier risk of a distorted teacher.
- Its calibration curve for any coefficient distortion (Theorem 5.3).
- The residual risk after exact recalibration (Corollary 5.4). The recalibration map is a temperature scaling. It removes all excess risk if and only if the teacher coefficient is a multiple of the outcome coefficient.

**Candidate sets**

- A coherence condition for independently computed scores when an event is split into subevents. This is Luce's choice axiom.

**Simulations**

- Under Gaussian design, simulation agrees with every formula.
- With the same covariance but non-Gaussian features, simulated risk at n = 12 is up to 10% below the formulas (uniform) and up to 35% above them (multivariate t). At n = 95 both are within 7%.

## Checks for practitioners

The conclusion of the paper turns these results into five checks for candidate-scoring models:

1. **Identify the training target.** A model trained on teacher or reference probabilities estimates the teacher's distribution. Its error against outcomes includes a teacher bias that more student data does not remove (Proposition 3.2). Agreement with the teacher does not reveal that bias (Equation 21, Section 5.3).
2. **Evaluate on realized outcomes with proper scores.** Compare the model with a conventional classifier fitted to the same outcomes. Separately, compare it with one fitted to the same teacher probabilities. This separates the effect of the architecture from the effect of the training target.
3. **Treat a pretrained representation as a way to reduce estimation variance.** It lowers the risk only while the signal it discards is below the threshold in Corollary 4.2, and that threshold decreases as the sample grows.
4. **Treat calibration as a correction of scale.** A calibrated forecast can still carry the excess risk of a wrong coefficient direction (Corollary 5.4), and calibration diagnostics do not detect it.
5. **Test coherence across candidate sets.** Pose the same event under different candidate sets and check that the probabilities agree.

A model that predicts which candidate an annotator, a teacher or a past policy would select has not thereby been shown to select the action with the best outcome. That requires a model of outcomes given actions and, when actions change outcomes, causal assumptions.

## Open question

The paper gives the statistical tradeoffs but not an empirical comparison. The open question is how a candidate-scoring model compares with a frozen pretrained embedding followed by a conventional calibrated classifier or kernel method, when both are trained on the same observations and evaluated with proper scores on held-out outcomes over paired splits and seeds.

## Reproducing the experiments

The script uses PyTorch and runs on a GPU if one is available, otherwise on the CPU. The run reported in the paper used one NVIDIA GB10 and took a few minutes.

```bash
pip install -r requirements.txt
python experiments.py
```

The script writes `results.json` and `validation_sweeps.{pdf,png}` to the current directory and prints the main numbers. The seed (20261016) and all parameters are set at the top of the script and listed in Table 1 of the paper. With PyTorch 2.10, NumPy 2.2, SciPy 1.13 and Matplotlib 3.10 the output matches the numbers in the paper.

## Building the paper

```bash
pdflatex paper.tex && pdflatex paper.tex
```

The figure `validation_sweeps.pdf` must be in the same directory.

## Citation

```bibtex
@misc{sudjianto2026teacher,
  author = {Sudjianto, Agus and Burakov, Denis},
  title  = {Teacher Supervision, Fixed Representations and Calibration:
            Exact Risk Calculations for Candidate-Scoring Decision Models},
  year   = {2026},
  howpublished = {\url{https://github.com/asudjianto-xml/Jev-Analysis}}
}
```
