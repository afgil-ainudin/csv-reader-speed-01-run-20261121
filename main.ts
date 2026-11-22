// The program the harness runs: node /bot/main.ts (node22). It reads every
// case from /input/cases.json and writes /output/answers.json — as one
// transducer pass over the raw bytes (see solve.js), not by parsing the JSON
// and stringifying the answers. Outside a container MAKEFUN_INPUT_DIR and
// MAKEFUN_OUTPUT_DIR stand in for /input and /output.
//
// Round 6 (tier-up off): the WASM transducer parses each case's text
// directly in its JSON-escaped form — no unescape pass — and streams the
// answers JSON straight to the output while it scans (two-stage SIMD
// masks, no per-record scratch). Measured locally: transduce ~33 ms for
// the 6 practice cases (round 5: ~33 ms, round 4: ~36 ms, round 3: ~44 ms);
// practice 6/6 byte-identical, differential fuzz 3,128 checks, 0 mismatches.
//
// Why this round changes main.ts instead of the kernel: the kernel is at
// its practical floor (~30 ms local; three restructure rounds proved the
// per-field reader cannot go faster in Liftoff), but the SCORED time is
// the whole process's CPU, not the parse alone — and a fresh process
// pays for work the parse never uses. Measured on this machine, the
// first (and only) transduce call in a fresh process costs ~80-110 ms
// of CPU while its wall time is only ~50-65 ms: the extra CPU runs on
// background threads. Two suspects were eliminated by measurement —
// first-touch page faults are only ~15 ms (7,029 minor faults for the
// ~28 MB of input + output pages, ~2 µs each), and lazy Liftoff
// compilation is nil (a tiny first call costs 1.4 ms, and instantiation
// itself 1.9 ms). What remains is V8's tier-up: Liftoff code carries
// counters, and once the big parse loop exhausts its budget V8 queues
// the hot functions for TurboFan, which compiles them on background
// threads (see v8.dev "WebAssembly Dynamic Tiering": background TurboFan
// "can block other tasks that would require the CPU", and it exists to
// speed up code that runs again). Our graded run executes the parse
// exactly once per fresh process, so the optimized code can never be
// reused — the background compile is pure scored cost with zero benefit.
// Turning both tier-up switches off removes the gap entirely (first
// transduce CPU == wall time, ~65 ms, in the same measurement where the
// default flags show CPU ~25-30 ms above wall), and later calls are
// unchanged (~34-36 ms). The flags are set here, before solve.js is
// imported, exactly like --no-incremental-marking already was; the
// house baseline's driver sets its flags the same way, and the rules
// explicitly allow an entry to tune V8's flags too.
//
// Round 7 (output flush windows): the parse is at its floor, so this
// round attacks the other first-call cost — first-touch page faults.
// The whole-file path used to build all 12 MiB of answers in one output
// region (~12 MiB of fresh output pages on top of ~11 MiB of input
// pages); now the kernel hands each full 512 KiB window to the caller
// through flush_out as soon as a row completes, and reuses the same
// 2 MiB of pages for the rest. The kernel is otherwise unchanged, and
// the flags above are unchanged; the entry still parses once and exits.

import { join } from 'node:path';
import { setFlagsFromString } from 'node:v8';

// V8's incremental marking off, set before solve.js is imported so all of its
// work runs under it (the rules: the house baseline runs with
// --no-incremental-marking, and an entry may tune V8's flags too).
// --no-wasm-tier-up and --no-wasm-dynamic-tiering keep every function on
// Liftoff for the whole run: the graded process parses once and exits, so
// background TurboFan recompilation could never pay for itself — it only
// adds scored CPU (see the header note above for the measurement).
setFlagsFromString('--no-incremental-marking --no-wasm-tier-up --no-wasm-dynamic-tiering');

/** solve.js, imported by its URL (a .js module carries no types). */
const { transduceFile } = (await import(new URL('./solve.js', import.meta.url).href)) as {
  transduceFile: (inPath: string, outPath: string) => void;
};

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

transduceFile(join(INPUT_DIR, 'cases.json'), join(OUTPUT_DIR, 'answers.json'));
