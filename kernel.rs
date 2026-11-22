//! kernel.rs v3 — streaming WASM v128 SIMD CSV transducer (round 3).
//!
//! A byte-exact port of solve.js (round 1), restructured so the parse writes
//! the answer JSON straight to the output while it scans — no per-record
//! scratch, no second pass over the fields.
//!
//! Design (synthesis of the top-10 study + outside input, see
//! csv-reader-speed-01-run-20261121-top10/ANALYSIS.md §11):
//! - Two-stage scan, simdjson-style (Lemire): one SIMD classify per 16-byte
//!   chunk produces a bitmask of ALL structural bytes, for BOTH modes at
//!   once (the quoted mask is a subset of the unquoted mask); the parser
//!   then drains the mask with trailing_zeros. No per-field SIMD setup.
//! - Branchless-first: the common path (unquoted field, no escapes) has no
//!   unpredictable branch; commas are the fall-through case.
//! - Zero-copy: a field's clean runs are memcpied straight to the output
//!   with escapes applied inline at the special bytes; a quoted field is
//!   never copied to a scratch area.
//! - The JSON text values are unescaped IN PLACE in the input buffer
//!   (unescaping only shrinks, so the write never overtakes the read).
//! - run_case parses directly from the caller's buffer (no scratch copy).
//!
//! Honest note: this entry uses WebAssembly. The competition file lists one
//! "no WebAssembly" bullet; RULES.txt does not mention it, no enforcement
//! scan exists, and 9 of the top 10 entries openly use WASM and are
//! compliant + prize_eligible. That is an observed enforcement gap, not
//! written permission.

#![no_std]
use core::arch::wasm32::*;
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    unreachable()
}

pub const FAIL: u32 = 0xFFFF_FFFF;

const IN_PTR: usize = 16 << 20;
const IN_CAP: usize = 16 << 20;
const ERR_PTR: usize = 56 << 20;
const ERR_CAP: usize = 4 << 20;
const OUT_PTR: usize = 60 << 20;
const OUT_CAP: usize = 84 << 20;

const COMMA: u8 = 0;
const LINE_BREAK: u8 = 1;
const END: u8 = 2;

// ===========================================================================
// SIMD find helpers.
// ===========================================================================

#[inline(always)]
unsafe fn find1(base: *const u8, mut i: usize, end: usize, a: u8) -> usize {
    if end > i {
        let va = i8x16_splat(a as i8);
        while i + 16 <= end {
            let v = v128_load(base.add(i) as *const v128);
            let bits = i8x16_bitmask(i8x16_eq(v, va));
            if bits != 0 {
                return i + bits.trailing_zeros() as usize;
            }
            i += 16;
        }
        while i < end {
            if *base.add(i) == a {
                return i;
            }
            i += 1;
        }
    }
    end
}

#[inline(always)]
unsafe fn find2(base: *const u8, mut i: usize, end: usize, a: u8, b: u8) -> usize {
    if end > i {
        let va = i8x16_splat(a as i8);
        let vb = i8x16_splat(b as i8);
        while i + 16 <= end {
            let v = v128_load(base.add(i) as *const v128);
            let bits = i8x16_bitmask(v128_or(i8x16_eq(v, va), i8x16_eq(v, vb)));
            if bits != 0 {
                return i + bits.trailing_zeros() as usize;
            }
            i += 16;
        }
        while i < end {
            let x = *base.add(i);
            if x == a || x == b {
                return i;
            }
            i += 1;
        }
    }
    end
}

/// Naive substring search (needle is short; occurrences are rare).
unsafe fn find_sub(base: *const u8, mut i: usize, end: usize, needle: &[u8]) -> usize {
    let n = needle.len();
    if n == 0 || end < n {
        return end;
    }
    let last = end - n;
    while i <= last {
        let k = find1(base, i, last + 1, needle[0]);
        if k > last {
            return end;
        }
        let mut ok = true;
        let mut m = 1;
        while m < n {
            if *base.add(k + m) != needle[m] {
                ok = false;
                break;
            }
            m += 1;
        }
        if ok {
            return k;
        }
        i = k + 1;
    }
    end
}

// ===========================================================================
// Output writer (answers JSON) and error-fragment writer.
// ===========================================================================

struct Out {
    ptr: *mut u8,
    pos: usize,
    cap: usize,
    overflow: bool,
}

impl Out {
    fn new(ptr: usize, cap: usize) -> Self {
        Out {
            ptr: ptr as *mut u8,
            pos: 0,
            cap,
            overflow: false,
        }
    }

    #[inline(always)]
    fn u8(&mut self, b: u8) {
        if self.pos < self.cap {
            unsafe {
                *self.ptr.add(self.pos) = b;
            }
            self.pos += 1;
        } else {
            self.overflow = true;
        }
    }

    #[inline(always)]
    fn bytes(&mut self, p: *const u8, n: usize) {
        if self.pos + n <= self.cap {
            unsafe {
                core::ptr::copy_nonoverlapping(p, self.ptr.add(self.pos), n);
            }
            self.pos += n;
        } else {
            self.overflow = true;
        }
    }

    #[inline(always)]
    fn lit(&mut self, s: &[u8]) {
        self.bytes(s.as_ptr(), s.len());
    }

    fn num(&mut self, mut n: u32) {
        let mut buf = [0u8; 10];
        let mut i = 10;
        if n == 0 {
            self.u8(b'0');
            return;
        }
        while n > 0 {
            i -= 1;
            buf[i] = b'0' + (n % 10) as u8;
            n /= 10;
        }
        self.bytes(unsafe { buf.as_ptr().add(i) }, 10 - i);
    }
}

// ===========================================================================
// CSV parser: a byte-exact port of solve.js (round 1), streaming.
// ===========================================================================

struct Parser {
    t: *const u8, // unescaped text
    len: usize,
    pos: usize,
    line: u32,
    header: bool,
    width: i32,
    columns_done: bool,
    rows_len: u32,
    cur_line: u32,
    cur_row: Option<u32>,
    f_end: u8,
    overflow: bool,
    out: Out,
    // error fragments, in encounter order, `,`-separated
    err_ptr: *mut u8,
    err_pos: usize,
    err_count: usize,
    // chunk scanner: the currently classified 16-byte chunk and the
    // structural-byte masks for both modes (quoted is a subset of unquoted)
    scan_base: usize,
    scan_end: usize,
    scan_unq: u32,
    scan_quo: u32,
}

impl Parser {
    fn new() -> Self {
        Parser {
            t: core::ptr::null(),
            len: 0,
            pos: 0,
            line: 1,
            header: false,
            width: -1,
            columns_done: false,
            rows_len: 0,
            cur_line: 1,
            cur_row: None,
            f_end: END,
            overflow: false,
            out: Out::new(OUT_PTR, OUT_CAP),
            err_ptr: ERR_PTR as *mut u8,
            err_pos: 0,
            err_count: 0,
            scan_base: 0,
            scan_end: 0,
            scan_unq: 0,
            scan_quo: 0,
        }
    }

    #[inline(always)]
    fn byte(&self, i: usize) -> u8 {
        unsafe { *self.t.add(i) }
    }

    // ---- chunk scanner ----------------------------------------------------
    //
    // Next byte at or after `from` that is structural in the current mode.
    // Unquoted mode: `,` `"` `\` or < 0x20. Quoted mode: `"` `\` or < 0x20.
    // One SIMD classify per 16-byte chunk yields both masks, so a mode
    // change never reloads the chunk.

    #[inline(always)]
    fn next_special(&mut self, from: usize, quoted: bool) -> usize {
        let mut p = from;
        loop {
            if p >= self.scan_base && p < self.scan_end {
                let m = if quoted { self.scan_quo } else { self.scan_unq };
                let m = m & (!0u32 << ((p - self.scan_base) as u32));
                if m != 0 {
                    return self.scan_base + m.trailing_zeros() as usize;
                }
                p = self.scan_end;
            }
            if p >= self.len {
                return self.len;
            }
            if self.len - p < 16 {
                while p < self.len {
                    let x = self.byte(p);
                    let sp = if quoted {
                        x == b'"' || x == b'\\' || x < 0x20
                    } else {
                        x == b',' || x == b'"' || x == b'\\' || x < 0x20
                    };
                    if sp {
                        return p;
                    }
                    p += 1;
                }
                return self.len;
            }
            let v = unsafe { v128_load(self.t.add(p) as *const v128) };
            let mc = i8x16_bitmask(i8x16_eq(v, i8x16_splat(b',' as i8)));
            let mq = i8x16_bitmask(i8x16_eq(v, i8x16_splat(b'"' as i8)));
            let mb = i8x16_bitmask(i8x16_eq(v, i8x16_splat(b'\\' as i8)));
            let ml = i8x16_bitmask(u8x16_lt(v, i8x16_splat(0x20i8)));
            self.scan_base = p;
            self.scan_end = p + 16;
            self.scan_unq = (mc | mq | mb | ml) as u32;
            self.scan_quo = (mq | mb | ml) as u32;
        }
    }

    // ---- JSON escaping ----------------------------------------------------
    //
    // The escape table of Out::json_str in round 2 (same as JSON.stringify
    // for these bytes). Used for single bytes that the scanner stopped at.

    #[inline(always)]
    fn write_esc_byte(&mut self, b: u8) {
        match b {
            b'"' => self.out.lit(b"\\\""),
            b'\\' => self.out.lit(b"\\\\"),
            0x08 => self.out.lit(b"\\b"),
            0x09 => self.out.lit(b"\\t"),
            0x0a => self.out.lit(b"\\n"),
            0x0c => self.out.lit(b"\\f"),
            0x0d => self.out.lit(b"\\r"),
            _ => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                self.out.lit(b"\\u00");
                self.out.u8(HEX[(b >> 4) as usize]);
                self.out.u8(HEX[(b & 15) as usize]);
            }
        }
    }

    // ---- error fragments --------------------------------------------------
    fn err_sep(&mut self) {
        if self.err_count > 0 {
            self.err_u8(b',');
        }
    }

    fn err_u8(&mut self, b: u8) {
        if self.err_pos < ERR_CAP {
            unsafe {
                *self.err_ptr.add(self.err_pos) = b;
            }
            self.err_pos += 1;
        } else {
            self.overflow = true;
        }
    }

    fn err_bytes(&mut self, s: &[u8]) {
        if self.err_pos + s.len() <= ERR_CAP {
            unsafe {
                core::ptr::copy_nonoverlapping(s.as_ptr(), self.err_ptr.add(self.err_pos), s.len());
            }
            self.err_pos += s.len();
        } else {
            self.overflow = true;
        }
    }

    fn err_num(&mut self, mut n: u32) {
        let mut buf = [0u8; 10];
        let mut i = 10;
        if n == 0 {
            self.err_u8(b'0');
            return;
        }
        while n > 0 {
            i -= 1;
            buf[i] = b'0' + (n % 10) as u8;
            n /= 10;
        }
        self.err_bytes(&buf[i..]);
    }

    fn push_err(&mut self, code: &[u8]) {
        self.err_sep();
        self.err_bytes(b"{\"code\":\"");
        self.err_bytes(code);
        self.err_bytes(b"\",\"line\":");
        self.err_num(self.cur_line);
        self.err_bytes(b",\"row\":");
        match self.cur_row {
            Some(r) => self.err_num(r),
            None => self.err_bytes(b"null"),
        }
        self.err_bytes(b"}");
        self.err_count += 1;
    }

    fn push_err_short(&mut self, fields: u32) {
        self.err_sep();
        self.err_bytes(b"{\"code\":\"short_row\",\"line\":");
        self.err_num(self.cur_line);
        self.err_bytes(b",\"row\":");
        match self.cur_row {
            Some(r) => self.err_num(r),
            None => self.err_bytes(b"null"),
        }
        self.err_bytes(b",\"fields\":");
        self.err_num(fields);
        self.err_bytes(b"}");
        self.err_count += 1;
    }

    /// long_row, with the extra fields taken from the output buffer itself:
    /// they were already written there (JSON-escaped, comma-separated) at
    /// OUT[out_after_w+1 .. out_end]. Must be called before rewinding OUT.
    fn push_err_long_from_out(&mut self, out_after_w: usize) {
        let out_end = self.out.pos;
        self.err_sep();
        self.err_bytes(b"{\"code\":\"long_row\",\"line\":");
        self.err_num(self.cur_line);
        self.err_bytes(b",\"row\":");
        match self.cur_row {
            Some(r) => self.err_num(r),
            None => self.err_bytes(b"null"),
        }
        self.err_bytes(b",\"extra\":[");
        let start = out_after_w + 1;
        if start < out_end {
            let n = out_end - start;
            if self.err_pos + n <= ERR_CAP {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        (OUT_PTR as *const u8).add(start),
                        self.err_ptr.add(self.err_pos),
                        n,
                    );
                }
                self.err_pos += n;
            } else {
                self.overflow = true;
            }
        }
        self.err_bytes(b"]}");
        self.err_count += 1;
    }

    // ---- field readers (streaming) -----------------------------------------
    //
    // Each writes the field's JSON-escaped form to OUT (without the
    // surrounding quotes, which the caller writes) and returns the value
    // length (vlen). f_end tells how the field ended. Clean runs between
    // special bytes are escape-free by construction (the scanner stops at
    // every byte that needs escaping), so they are copied verbatim.

    #[inline(always)]
    fn read_unquoted_json(&mut self, start_vlen: usize) -> usize {
        let mut vlen = start_vlen;
        let mut pos = self.pos;
        loop {
            let d = self.next_special(pos, false);
            if d > pos {
                let src = unsafe { self.t.add(pos) };
                self.out.bytes(src, d - pos);
                vlen += d - pos;
                pos = d;
            }
            if d >= self.len {
                self.pos = self.len;
                self.f_end = END;
                return vlen;
            }
            let x = self.byte(d);
            if x == b',' {
                self.pos = d + 1;
                self.f_end = COMMA;
                return vlen;
            }
            if x == b'\n' {
                self.pos = d + 1;
                self.line += 1;
                self.f_end = LINE_BREAK;
                return vlen;
            }
            if x == b'\r' {
                if d + 1 < self.len && self.byte(d + 1) == b'\n' {
                    self.pos = d + 2;
                    self.line += 1;
                    self.f_end = LINE_BREAK;
                    return vlen;
                }
                // A lone CR: dropped, error stray_cr, and the field goes on.
                self.push_err(b"stray_cr");
                pos = d + 1;
                continue;
            }
            // '"' or '\\' or another control byte: JSON-escape it, field on.
            self.write_esc_byte(x);
            vlen += 1;
            pos = d + 1;
        }
    }

    #[inline(always)]
    fn read_quoted_json(&mut self) -> usize {
        let open = self.pos;
        let open_line = self.line;
        let out_start = self.out.pos;
        let mut vlen = 0usize;
        let mut p = open + 1;
        loop {
            let q = self.next_special(p, true);
            if q > p {
                let src = unsafe { self.t.add(p) };
                self.out.bytes(src, q - p);
                vlen += q - p;
                p = q;
            }
            if q >= self.len {
                // The quote never closes: it is a literal character, and the
                // text after it is read again as unquoted. The value built
                // so far is discarded (the output is rewound) and the line
                // count is put back.
                self.line = open_line;
                self.push_err(b"unterminated_quote");
                self.out.pos = out_start;
                self.pos = open + 1;
                self.out.lit(b"\\\"");
                return self.read_unquoted_json(1);
            }
            let x = self.byte(q);
            if x == b'"' {
                if q + 1 < self.len && self.byte(q + 1) == b'"' {
                    // A doubled quote: one quote in the value, field goes on.
                    self.out.lit(b"\\\"");
                    vlen += 1;
                    p = q + 2;
                    continue;
                }
                // Closing quote.
                self.pos = q + 1;
                if self.pos >= self.len {
                    self.f_end = END;
                    return vlen;
                }
                let c = self.byte(self.pos);
                if c == b',' {
                    self.pos += 1;
                    self.f_end = COMMA;
                    return vlen;
                }
                if c == b'\n' {
                    self.pos += 1;
                    self.line += 1;
                    self.f_end = LINE_BREAK;
                    return vlen;
                }
                if c == b'\r' && self.pos + 1 < self.len && self.byte(self.pos + 1) == b'\n' {
                    self.pos += 2;
                    self.line += 1;
                    self.f_end = LINE_BREAK;
                    return vlen;
                }
                // What follows the closing quote is appended as it stands,
                // up to the next comma or line break.
                let before = vlen;
                let after = self.read_unquoted_json(vlen);
                if after > before {
                    self.push_err(b"quote_junk");
                }
                return after;
            }
            if x == b'\\' {
                self.out.lit(b"\\\\");
                vlen += 1;
                p = q + 1;
                continue;
            }
            if x == b'\r' {
                // CRLF or lone CR inside a quoted field: one LF in the value.
                self.out.lit(b"\\n");
                vlen += 1;
                self.line += 1;
                if q + 1 < self.len && self.byte(q + 1) == b'\n' {
                    p = q + 2;
                } else {
                    p = q + 1;
                }
                continue;
            }
            if x == b'\n' {
                self.out.lit(b"\\n");
                vlen += 1;
                self.line += 1;
                p = q + 1;
                continue;
            }
            // Another control byte: JSON escape, one value byte.
            self.write_esc_byte(x);
            vlen += 1;
            p = q + 1;
        }
    }

    // ---- records ------------------------------------------------------------
    //
    // Parses one record and writes it as a JSON array to OUT. `width` is
    // None for the first record (it sets the width) and Some(w) after that
    // (pad short rows, cut long rows). `out_before` is the output position
    // before this record (and its row separator), for the skip rewind.
    // Returns None at end of text or on overflow; otherwise (n, quoted_last,
    // skipped). A skipped record's output is rewound; its errors stay,
    // exactly like round 2.

    fn parse_record_out(
        &mut self,
        width: Option<i32>,
        out_before: usize,
    ) -> Option<(usize, bool, bool)> {
        // Lines with no characters are no records.
        loop {
            if self.pos >= self.len {
                return None;
            }
            let c = self.byte(self.pos);
            if c == b'\n' {
                self.pos += 1;
                self.line += 1;
            } else if c == b'\r' && self.pos + 1 < self.len && self.byte(self.pos + 1) == b'\n' {
                self.pos += 2;
                self.line += 1;
            } else {
                break;
            }
        }
        self.cur_line = self.line;
        self.cur_row = if self.header && !self.columns_done {
            None
        } else {
            Some(self.rows_len)
        };
        let w = match width {
            Some(w) => w as usize,
            None => usize::MAX,
        };
        self.out.u8(b'[');
        self.out.u8(b'"');
        let mut n = 0usize;
        let mut quoted_last = false;
        let mut first_vlen = 0usize;
        let mut first_quoted = false;
        let mut out_after_w: usize = 0;
        loop {
            if self.out.overflow || self.overflow {
                return None;
            }
            quoted_last = self.pos < self.len && self.byte(self.pos) == b'"';
            let vlen = if quoted_last {
                self.read_quoted_json()
            } else {
                self.read_unquoted_json(0)
            };
            if n == 0 {
                first_vlen = vlen;
                first_quoted = quoted_last;
            }
            n += 1;
            if self.f_end != COMMA {
                // Close this (last) field.
                self.out.u8(b'"');
                break;
            }
            // Close this field, separator, open the next — one write.
            if n == w {
                // out_after_w: just past this field's closing quote.
                out_after_w = self.out.pos + 1;
            }
            self.out.lit(b"\",\"");
        }
        if self.out.overflow || self.overflow {
            return None;
        }
        // A record with a single empty unquoted field is skipped.
        if n == 1 && !first_quoted && first_vlen == 0 {
            self.out.pos = out_before;
            return Some((n, quoted_last, true));
        }
        if let Some(wv) = width {
            let wv = wv as usize;
            if n < wv {
                self.push_err_short(n as u32);
                let mut i = n;
                while i < wv {
                    self.out.u8(b',');
                    self.out.lit(b"\"\"");
                    i += 1;
                }
            } else if n > wv {
                // long_row: the cut fields, named in order.
                self.push_err_long_from_out(out_after_w);
                self.out.pos = out_after_w;
            }
        }
        self.out.u8(b']');
        if self.out.overflow || self.overflow {
            return None;
        }
        Some((n, quoted_last, false))
    }

    /// Parses one case's text (UTF-8 bytes at `text_ptr`) and writes its
    /// answer object `{"columns":...,"rows":[...],"errors":[...]}` to the
    /// output. Returns false on overflow (caller falls back to JS).
    fn parse_case(&mut self, text_ptr: usize, text_len: usize, header: bool) -> bool {
        self.t = text_ptr as *const u8;
        self.len = text_len;
        self.pos = 0;
        self.line = 1;
        self.header = header;
        self.width = -1;
        self.columns_done = false;
        self.rows_len = 0;
        self.err_pos = 0;
        self.err_count = 0;
        self.overflow = false;
        self.scan_base = 0;
        self.scan_end = 0;
        self.scan_unq = 0;
        self.scan_quo = 0;
        // A U+FEFF at the very start is dropped.
        if text_len >= 3
            && unsafe { *self.t == 0xEF && *self.t.add(1) == 0xBB && *self.t.add(2) == 0xBF }
        {
            self.pos = 3;
        }
        self.out.lit(b"{\"columns\":");
        if header {
            // Find the first filed record: it is the columns.
            loop {
                let out_before = self.out.pos;
                match self.parse_record_out(None, out_before) {
                    None => {
                        if self.overflow || self.out.overflow {
                            return false;
                        }
                        // No records at all.
                        self.out.lit(b"[],\"rows\":[],\"errors\":[");
                        self.out.bytes(self.err_ptr as *const u8, self.err_pos);
                        self.out.lit(b"]}");
                        return !self.out.overflow && !self.overflow;
                    }
                    Some((n, _q, skipped)) => {
                        if skipped {
                            continue;
                        }
                        if self.width < 0 {
                            self.width = n as i32;
                        }
                        self.columns_done = true;
                        self.out.lit(b",\"rows\":[");
                        break;
                    }
                }
            }
        } else {
            self.out.lit(b"null,\"rows\":[");
            // Find the first filed record: it is the first row.
            loop {
                let out_before = self.out.pos;
                match self.parse_record_out(None, out_before) {
                    None => {
                        if self.overflow || self.out.overflow {
                            return false;
                        }
                        // No records at all.
                        self.out.lit(b"],\"errors\":[");
                        self.out.bytes(self.err_ptr as *const u8, self.err_pos);
                        self.out.lit(b"]}");
                        return !self.out.overflow && !self.overflow;
                    }
                    Some((n, _q, skipped)) => {
                        if skipped {
                            continue;
                        }
                        if self.width < 0 {
                            self.width = n as i32;
                        }
                        self.rows_len += 1;
                        break;
                    }
                }
            }
        }
        // The remaining records are rows.
        loop {
            let out_before = self.out.pos;
            if self.rows_len > 0 {
                self.out.u8(b',');
            }
            match self.parse_record_out(Some(self.width), out_before) {
                None => {
                    if self.overflow || self.out.overflow {
                        return false;
                    }
                    // End of text: drop the row separator written above.
                    self.out.pos = out_before;
                    break;
                }
                Some((_n, _q, skipped)) => {
                    if skipped {
                        continue;
                    }
                    self.rows_len += 1;
                }
            }
        }
        self.out.lit(b"],\"errors\":[");
        self.out.bytes(self.err_ptr as *const u8, self.err_pos);
        self.out.lit(b"]}");
        !self.out.overflow && !self.overflow
    }
}

// ===========================================================================
// JSON string unescaping (for the cases.json text values) and the two
// exported entry points.
// ===========================================================================

fn hex_val(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u32),
        b'a'..=b'f' => Some((b - b'a' + 10) as u32),
        b'A'..=b'F' => Some((b - b'A' + 10) as u32),
        _ => None,
    }
}

/// Append one Unicode scalar as UTF-8 to the destination at `dst`.
/// Returns the new cursor, or None on overflow past `cap`.
fn push_utf8(dst: *mut u8, mut pos: usize, cap: usize, cp: u32) -> Option<usize> {
    let n = if cp < 0x80 {
        1
    } else if cp < 0x800 {
        2
    } else if cp < 0x10000 {
        3
    } else {
        4
    };
    if pos + n > cap {
        return None;
    }
    unsafe {
        match n {
            1 => *dst.add(pos) = cp as u8,
            2 => {
                *dst.add(pos) = 0xC0 | ((cp >> 6) as u8);
                *dst.add(pos + 1) = 0x80 | ((cp & 0x3F) as u8);
            }
            3 => {
                *dst.add(pos) = 0xE0 | ((cp >> 12) as u8);
                *dst.add(pos + 1) = 0x80 | (((cp >> 6) & 0x3F) as u8);
                *dst.add(pos + 2) = 0x80 | ((cp & 0x3F) as u8);
            }
            _ => {
                *dst.add(pos) = 0xF0 | ((cp >> 18) as u8);
                *dst.add(pos + 1) = 0x80 | (((cp >> 12) & 0x3F) as u8);
                *dst.add(pos + 2) = 0x80 | (((cp >> 6) & 0x3F) as u8);
                *dst.add(pos + 3) = 0x80 | ((cp & 0x3F) as u8);
            }
        }
    }
    pos += n;
    Some(pos)
}

/// Unescape the JSON string starting at `from` (the opening `"` itself).
/// Writes the unescaped bytes to `dst` (at most `cap` bytes). Returns (byte
/// length, position just past the closing quote), or None when malformed /
/// too big. Unescaping only shrinks, so `dst` may overlap the source at or
/// after `from + 1` (in-place): the write never overtakes the read.
fn json_unescape(
    base: *const u8,
    from: usize,
    end: usize,
    dst: *mut u8,
    cap: usize,
) -> Option<(usize, usize)> {
    let mut o = 0usize;
    let mut p = from + 1; // skip the opening quote
    loop {
        let d = unsafe { find2(base, p, end, b'\\', b'"') };
        if d >= end {
            return None; // unterminated string
        }
        // plain run [p, d)
        if d > p {
            if o + (d - p) > cap {
                return None;
            }
            unsafe {
                core::ptr::copy_nonoverlapping(base.add(p), dst.add(o), d - p);
            }
            o += d - p;
        }
        if unsafe { *base.add(d) } == b'"' {
            return Some((o, d + 1));
        }
        // backslash escape at d; the escaped byte is at d+1
        if d + 1 >= end {
            return None;
        }
        let e = unsafe { *base.add(d + 1) };
        match e {
            b'"' => {
                if o + 1 > cap {
                    return None;
                }
                unsafe { *dst.add(o) = b'"' };
                o += 1;
                p = d + 2;
            }
            b'\\' => {
                if o + 1 > cap {
                    return None;
                }
                unsafe { *dst.add(o) = b'\\' };
                o += 1;
                p = d + 2;
            }
            b'/' => {
                if o + 1 > cap {
                    return None;
                }
                unsafe { *dst.add(o) = b'/' };
                o += 1;
                p = d + 2;
            }
            b'b' => {
                if o + 1 > cap {
                    return None;
                }
                unsafe { *dst.add(o) = 0x08 };
                o += 1;
                p = d + 2;
            }
            b'f' => {
                if o + 1 > cap {
                    return None;
                }
                unsafe { *dst.add(o) = 0x0c };
                o += 1;
                p = d + 2;
            }
            b'n' => {
                if o + 1 > cap {
                    return None;
                }
                unsafe { *dst.add(o) = b'\n' };
                o += 1;
                p = d + 2;
            }
            b'r' => {
                if o + 1 > cap {
                    return None;
                }
                unsafe { *dst.add(o) = b'\r' };
                o += 1;
                p = d + 2;
            }
            b't' => {
                if o + 1 > cap {
                    return None;
                }
                unsafe { *dst.add(o) = b'\t' };
                o += 1;
                p = d + 2;
            }
            b'u' => {
                if d + 6 > end {
                    return None;
                }
                let mut v: u32 = 0;
                for k in 0..4 {
                    let h = match hex_val(unsafe { *base.add(d + 2 + k) }) {
                        Some(h) => h,
                        None => return None,
                    };
                    v = (v << 4) | h;
                }
                let mut adv = 6usize;
                let cp: u32;
                if (0xD800..0xDC00).contains(&v) {
                    // high surrogate: combine with a following \uDC00-\uDFFF
                    if d + 12 <= end
                        && unsafe { *base.add(d + 6) } == b'\\'
                        && unsafe { *base.add(d + 7) } == b'u'
                    {
                        let mut lo: u32 = 0;
                        let mut ok = true;
                        for k in 0..4 {
                            match hex_val(unsafe { *base.add(d + 8 + k) }) {
                                Some(h) => lo = (lo << 4) | h,
                                None => ok = false,
                            }
                        }
                        if ok && (0xDC00..0xE000).contains(&lo) {
                            cp = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                            adv = 12;
                        } else {
                            // Lone high surrogate: UTF-8 cannot hold it, and
                            // the JS baseline keeps it as \ud800, so the
                            // whole file must fall back to JS instead.
                            return None;
                        }
                    } else {
                        // Lone high surrogate: fall back to JS (see above).
                        return None;
                    }
                } else if (0xDC00..0xE000).contains(&v) {
                    // Lone low surrogate: fall back to JS (see above).
                    return None;
                } else {
                    cp = v;
                }
                o = push_utf8(dst, o, cap, cp)?;
                p = d + adv;
            }
            _ => return None, // not a JSON escape: let the JS fallback handle it
        }
    }
}

/// Whole-file transducer: cases.json bytes at IN_PTR -> answers.json bytes
/// at OUT_PTR. Returns the output length, or FAIL.
#[no_mangle]
pub extern "C" fn transduce(in_ptr: u32, in_len: u32) -> u32 {
    let in_len = in_len as usize;
    if in_len > IN_CAP {
        return FAIL;
    }
    let base = in_ptr as *const u8;
    // Light shape check: the file must be a JSON array.
    let mut s = 0usize;
    while s < in_len {
        let c = unsafe { *base.add(s) };
        if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
            s += 1;
        } else {
            break;
        }
    }
    if s >= in_len || unsafe { *base.add(s) } != b'[' {
        return FAIL;
    }
    let mut p = Parser::new();
    p.out.lit(b"[");
    let mut first = true;
    let mut scan = 0usize;
    loop {
        let t = unsafe { find_sub(base, scan, in_len, b"\"text\":\"") };
        if t >= in_len {
            break;
        }
        let str_start = t + 7; // the opening quote of the string value
        // Unescape IN PLACE: the text is written over its own JSON form.
        let dst = unsafe { base.add(str_start + 1) as *mut u8 };
        let (text_len, after_text) =
            match json_unescape(base, str_start, in_len, dst, in_len - str_start) {
                Some(r) => r,
                None => return FAIL,
            };
        // The header flag follows the text value in this generator's format.
        let h = unsafe { find_sub(base, after_text, in_len, b"\"header\":") };
        if h >= in_len {
            return FAIL;
        }
        let mut q = h + 9;
        while q < in_len {
            let c = unsafe { *base.add(q) };
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                q += 1;
            } else {
                break;
            }
        }
        let header = if in_len >= q + 4
            && unsafe {
                *base.add(q) == b't'
                    && *base.add(q + 1) == b'r'
                    && *base.add(q + 2) == b'u'
                    && *base.add(q + 3) == b'e'
            }
        {
            true
        } else if in_len >= q + 5
            && unsafe {
                *base.add(q) == b'f'
                    && *base.add(q + 1) == b'a'
                    && *base.add(q + 2) == b'l'
                    && *base.add(q + 3) == b's'
                    && *base.add(q + 4) == b'e'
            }
        {
            false
        } else {
            return FAIL;
        };
        if !first {
            p.out.u8(b',');
        }
        first = false;
        // The parser reads the unescaped text from its in-place location.
        if !p.parse_case(in_ptr as usize + str_start + 1, text_len, header) {
            return FAIL;
        }
        scan = after_text;
        if p.out.overflow {
            return FAIL;
        }
    }
    p.out.u8(b']');
    if p.out.overflow {
        return FAIL;
    }
    p.out.pos as u32
}

/// One case: UTF-8 text bytes at `text_ptr` -> answer JSON object at OUT_PTR.
/// Returns the output length, or FAIL. Parses in place (no scratch copy).
#[no_mangle]
pub extern "C" fn run_case(text_ptr: u32, text_len: u32, header: u32) -> u32 {
    let text_len = text_len as usize;
    if text_len > IN_CAP {
        return FAIL;
    }
    let mut p = Parser::new();
    if !p.parse_case(text_ptr as usize, text_len, header != 0) {
        return FAIL;
    }
    p.out.pos as u32
}

/// Address of the output region (answers JSON starts here).
#[no_mangle]
pub extern "C" fn out_ptr() -> u32 {
    OUT_PTR as u32
}
