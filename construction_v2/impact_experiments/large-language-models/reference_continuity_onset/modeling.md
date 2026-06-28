# Weak Signal Impact Modeling Summary

The solution-space and problem-space formulas are intentionally different because they represent different types of weak signals. Solution-space topics are usually methods or techniques, so a good signal should look like a relatively stable adoption curve. The solution formula is therefore stricter about exponential shape, nonzero support, and whether the signal stays strong through 2023. Problem-space topics are often sparser and can appear later as direct precursors to a 2024 mature topic, so the problem formula keeps the exponential-fit requirement but uses a lighter fit score and gives 2024 reference validation a direct role in the final score. In short, solution-space is modeled as the gradual adoption of methods, while problem-space is modeled as direct precursor evidence.

## Solution-Space

Final script:

`scripts/score_solution_exponential_fit_quality_peak_2024_gate.py`

Final output folder:

`outputs_solution_exponential_fit_quality_peak_2024_gate/solution-space`

### Formula

For each candidate topic, the model tries onset years from 2019 to 2022. For a chosen onset year `s`, the fitted window is:

`topic_f(s), topic_f(s+1), ..., topic_f(2023)`

The exponential fit is:

`log(topic_f(t) + epsilon) = alpha + beta * t`

For each onset window, the raw score is:

`raw_onset_score = R2^3 * growth_reward * positive_share * peak_at_end * terminal * nonzero_penalty * pre_onset_penalty^2`

The best onset is the one with the highest `raw_onset_score`.

The final solution-space score is:

`impact_solution_fit_quality = normalized(best_raw_onset_score)`

### Metric Meaning

- `R2`: how close the log-frequency curve is to an exponential trend. This rewards curves that are not only increasing, but close to the expected exponential shape.
- `growth_reward`: `log(growth)` if `growth > 1`, otherwise 0. Here, `growth` is the 2023 frequency divided by the first nonzero frequency in the onset window. This rewards the magnitude of growth and prevents topics with a good exponential fit but almost no real increase from receiving a high score.
- `positive_share`: the share of year-to-year steps that increase. This prefers signals that rise across multiple steps rather than only jump once.
- `peak_at_end`: whether 2023 is close to the highest value in the whole onset window. This avoids selecting topics whose main peak happened before 2023.
- `terminal`: whether 2023 is close to 2022. This penalizes a sharp final-year drop.
- `nonzero_penalty`: penalty for windows with too few nonzero years. This reduces one-point or two-point noise.
- `pre_onset_penalty`: penalty for choosing an onset after an earlier peak. This prevents the model from ignoring a high pre-onset value.

### Hard Gate

The final solution-space top results must satisfy:

`impact_solution_fit_quality > 0`

and:

`ref_f_2024 >= max(topic_f_2019, topic_f_2020, topic_f_2021, topic_f_2022, topic_f_2023)`

Here, `ref_f_2024` is reference-based validation from 2024 non-survey papers, while `topic_f_2019` to `topic_f_2023` are survey-excluded topic frequencies.

### Current Result

Top 10:

1. retrieval-augmented language models
2. in-context learning
3. parameter-efficient tuning
4. few-shot learning with language models
5. autoregressive language models
6. language model pretraining
7. parameter-efficient transfer learning
8. retrieval-augmented question answering
9. vision-language pretraining
10. long-context language models

Top 10 table:

`outputs_solution_exponential_fit_quality_peak_2024_gate/solution-space/top10_solution_exponential_fit_quality_solution_space.md`

Top 10 figures:

`outputs_solution_exponential_fit_quality_peak_2024_gate/solution-space/top10_frequency_individual`

## Problem-Space

Final script:

`scripts/score_problem_exponential_fit_direct_precursor.py`

Final output folder:

`outputs_problem_exponential_fit_direct_precursor_v4_peak_2024_gate/problem-space`

### Formula

For each candidate topic, the model tries onset years from 2019 to 2022. For a chosen onset year `s`, the fitted window is:

`topic_f(s), topic_f(s+1), ..., topic_f(2023)`

The exponential fit is:

`log(topic_f(t) + epsilon) = alpha + beta * t`

For each onset window, the raw score is:

`problem_exp_raw = R2^2 * growth_reward * positive_share * end_strength * pre_onset_penalty`

The best onset is the one with the highest `problem_exp_raw`.

The final problem-space score is:

`impact_problem_exponential = fit_score * sqrt(validation_score)`

where:

- `fit_score = normalized(best_problem_exp_raw)`
- `validation_score = normalized(ref_f_2024)`

### Metric Meaning

- `R2`: how close the log-frequency curve is to an exponential trend. This keeps the score tied to exponential-growth shape.
- `growth_reward`: `log(growth)` if `growth > 1`, otherwise 0. This requires the topic to grow within the selected onset window.
- `positive_share`: the share of year-to-year steps that increase. This rewards multi-step growth rather than a single jump.
- `end_strength`: whether 2023 is close to the maximum value in the onset window. This penalizes topics that peaked before 2023.
- `pre_onset_penalty`: penalty for choosing an onset after an earlier peak. This avoids treating an old topic as newly emerging by starting the fit too late.
- `fit_score`: normalized exponential-fit evidence.
- `validation_score`: normalized 2024 reference evidence.

The problem-space score uses `fit_score * sqrt(validation_score)` so that exponential fit remains the main signal, while 2024 reference validation still contributes as confirmation.

### Hard Gate

The final problem-space top results must satisfy:

`impact_problem_exponential > 0`

and:

`ref_f_2024 >= max(topic_f_2019, topic_f_2020, topic_f_2021, topic_f_2022, topic_f_2023)`

and:

`topic_f_2019 <= topic_f_2023`

The first gate requires 2024 reference validation to be stronger than the entire 2019-2023 topic-frequency peak. The second gate removes topics that were already stronger in 2019 than in 2023.

### Current Result

Top 10:

1. privacy leakage in NLP models
2. machine-generated text detection
3. domain adaptation of pre-trained language models
4. pre-trained language model compression
5. efficient deployment of pre-trained language models
6. zero-shot task generalization
7. in-context learning mechanisms
8. efficient inference for pretrained language models
9. language model alignment
10. knowledge-based visual question answering

Top 10 table:

`outputs_problem_exponential_fit_direct_precursor_v4_peak_2024_gate/problem-space/top10_problem_exponential_fit_problem_space.md`

Top 10 figures:

`outputs_problem_exponential_fit_direct_precursor_v4_peak_2024_gate/problem-space/top10_frequency_individual`
