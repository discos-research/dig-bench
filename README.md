# DiG-bench — Discovery in Games

DiG-bench is a benchmark of scientific discovery: 70 text-based games, each with unknown
transformation rules that an agent must uncover through interaction and experimentation.
21 games are publicly released (`P-1`–`P-21`); the rest are held private for secure
evaluation. Every game has been beaten by at least one human on their first attempt.

- **Website, leaderboard, and API docs:** https://digbench.ai
- **Technical report:** [`tech_report.pdf`](tech_report.pdf).

## In this repository

- [`baseline-harness/`](baseline-harness/) — the minimal multi-provider agent behind the
  report's baseline-harness results: one agent that plays DiG-bench games across Gemini,
  Anthropic, OpenAI, and SGLang-served open models, with each model's reasoning carried
  across turns. See its [README](baseline-harness/README.md) for setup and usage, and
  [`baseline-harness/CITATION.cff`](baseline-harness/CITATION.cff) to cite it.

## Citation

```bibtex
@misc{battleday2026dig,
  title={DiG-bench: Discovery in Games},
  author={Ruairidh M. Battleday and Kai Sandbrink and Jimi Cullen-Drohan and Zihan Yan and 
  Timothy Muller and Clare Maguire and Ales Kubicek and Fraser Greenlee-Scott and 
  Sukrit Sumant and Tri Dao and Jürgen Schmidhuber and Michal Valko and 
  Joshua Tenenbaum and Thomas L. Griffiths and Zeb Kurth-Nelson and James C.R. Whittington},
  year={2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
