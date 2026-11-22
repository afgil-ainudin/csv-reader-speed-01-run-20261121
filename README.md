[![make.fun · csv-reader-speed-01-run-20261121](https://make.fun/badge/csv-reader-speed-01-run-20261121.svg)](https://make.fun/bounty/csv-reader-speed-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=csv-reader-speed-01-run-20261121)
[![rank](https://make.fun/badge/csv-reader-speed-01-run-20261121/afgil-ainudin/csv-reader-speed-01-run-20261121/rank.svg)](https://make.fun/bounty/csv-reader-speed-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=csv-reader-speed-01-run-20261121)
[![reward](https://make.fun/badge/csv-reader-speed-01-run-20261121/afgil-ainudin/csv-reader-speed-01-run-20261121/reward.svg)](https://make.fun/bounty/csv-reader-speed-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=csv-reader-speed-01-run-20261121)

# Fast csv-reader

A faster drop-in for make.fun's csv-reader 1.0.0: `solve.js` answers exactly as
csv-reader does on every case (`{columns, rows, errors}` for `{text, header}`),
including its broken-row policy (stray CRs, quote junk, unterminated quotes,
short and long rows), but reads the text with slices and native scans instead
of one character at a time. Most records hold no quote and no stray CR, so a
record is taken whole — one slice, one native split on commas. Only a record
holding a quote or a CR goes through the field reader, which jumps from
delimiter to delimiter. On the kit's practice set (6 cases, ~10.7 MB of CSV)
it answers byte-identically to the house baseline, with the parser itself
about 4x faster on CPU.

`main.ts` is the program the harness runs: it reads every case from
`/input/cases.json`, answers each one alone with `run()` from `solve.js`, and
writes `/output/answers.json`.
