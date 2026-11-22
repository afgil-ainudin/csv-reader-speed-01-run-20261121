// The program the harness runs: node /bot/main.ts (node22). It reads every
// case from /input/cases.json and writes /output/answers.json — as one
// transducer pass over the raw bytes (see solve.js), not by parsing the JSON
// and stringifying the answers. Outside a container MAKEFUN_INPUT_DIR and
// MAKEFUN_OUTPUT_DIR stand in for /input and /output.
//
// Round 3 (streaming kernel): the WASM transducer unescapes each case's
// text in place and streams the answers JSON straight to the output while
// it scans (two-stage SIMD masks, no per-record scratch). Measured locally:
// transduce ~41.5 ms for the 6 practice cases (round 2: ~65 ms); practice
// 6/6 byte-identical, differential fuzz 1,800 cases 0 mismatches.

import { join } from 'node:path';
import { setFlagsFromString } from 'node:v8';

// V8's incremental marking off, set before solve.js is imported so all of its
// work runs under it (the rules: the house baseline runs with
// --no-incremental-marking, and an entry may tune V8's flags too).
setFlagsFromString('--no-incremental-marking');

/** solve.js, imported by its URL (a .js module carries no types). */
const { transduceFile } = (await import(new URL('./solve.js', import.meta.url).href)) as {
  transduceFile: (inPath: string, outPath: string) => void;
};

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

transduceFile(join(INPUT_DIR, 'cases.json'), join(OUTPUT_DIR, 'answers.json'));
