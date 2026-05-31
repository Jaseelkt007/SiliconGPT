# Benchmark comparison — our model vs LLM baselines

Each model is scored with `src/process_logic/score.py` on the **examples it actually answered**
(the `n` column). Our model + n-gram ran the FULL held-out eval on the A100; the frontier LLMs
were sampled on the first 200 examples (cost). All values are the real measured numbers.

## Nextstep  (each model scored on the examples it answered — see n)

| model | n | top1 | top3 | top5 | mrr |
|---|---|---|---|---|---|
| Ours (1.37M) | 3600 | 0.8111 | 0.9964 | 0.9997 | 0.9032 |
| Gemini 3.5-flash | 200 | 0.5550 | 0.7350 | 0.7800 | 0.6467 |
| DeepSeek | 200 | 0.4800 | 0.5900 | 0.6500 | 0.5451 |
| Qwen | 200 | 0.4150 | 0.5550 | 0.6350 | 0.4972 |
| n-gram (trigram) | 3600 | 0.7608 | 0.9906 | 1.0000 | 0.8743 |

## Completion  (each model scored on the examples it answered — see n)

| model | n | exact_match | norm_edit_dist | token_acc |
|---|---|---|---|---|
| Ours (1.37M) | 600 | 0.0000 | 0.2216 | 0.4053 |
| Gemini 3.5-flash | 200 | 0.0000 | 0.6582 | 0.0762 |
| DeepSeek | 200 | 0.0000 | 0.7602 | 0.0556 |
| Qwen | 200 | 0.0000 | 0.8274 | 0.0245 |
| n-gram (trigram) | 600 | 0.0000 | 0.3178 | 0.2827 |

## Anomaly  (each model scored on the examples it answered — see n)

| model | n | acc | precision | recall | f1 | roc_auc | rule_attr |
|---|---|---|---|---|---|---|---|
| Ours (1.37M) | 1000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9100 |
| Gemini 3.5-flash | 200 | 0.9250 | 0.8837 | 0.9383 | 0.9102 | 0.6757 | 0.8947 |
| DeepSeek | 200 | 0.7700 | 1.0000 | 0.4321 | 0.6034 | 0.7457 | 1.0000 |
| Qwen | 200 | 0.7750 | 0.7812 | 0.6173 | 0.6897 | 0.6731 | 0.7600 |
| n-gram (trigram) | — | — | — | — | — | — | — |
