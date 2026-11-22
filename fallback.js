// Fast drop-in for csv-reader 1.0.0: run(input) returns exactly csv-reader's
// {columns, rows, errors} for {text, header}, but reads the text with slices and
// native scans instead of one character at a time.
//
// How it stays identical: it follows the same written policy, in the same order.
// A U+FEFF at the very start is dropped. A record ends at LF or CRLF, and a line
// left with no characters is no record. Outside quotes a CR not followed by LF is
// dropped (stray_cr). A field that opens with a quote runs to the next quote that
// is not doubled, "" inside it one quote and each CRLF or lone CR inside it read
// as LF; what follows its closing quote, up to the next comma or line break, is
// appended as it stands (quote_junk, when the field ends); a quote that never
// closes is a literal character, and the text after it is read again as unquoted
// (unterminated_quote). A quote inside an unquoted field is a literal character.
// Fields stay strings, never trimmed or converted. columns is the first record
// when header is true, else null; rows holds the records after it, each padded
// (short_row, with fields: its field count) or cut (long_row, with extra: the cut
// fields) to the first record's width. Each error is {code, line, row}, in the
// order met: line the 1-based line its record starts on, row its record's index
// in rows, null for the header.
//
// How it is fast: most records hold no quote and no stray CR, so a record is
// taken whole — one slice, one native split on commas — instead of a field at a
// time. Only a record that holds a quote or a CR goes through the field reader,
// which jumps from delimiter to delimiter with one native regex scan instead of
// stepping a character at a time. Fields are slices of the input, so no field is
// ever built up a character at a time.

const COMMA = 0;
const LINE_BREAK = 1;
const END = 2;

/** csv-reader 1.0.0's answer for one case's input, read fast. */
export function run(input) {
  const text = String(input.text);
  const header = input.header === true;
  const len = text.length;
  const errors = [];
  const rows = [];
  let columns = null;
  let width = -1;

  let pos = 0; // the next character to read
  let line = 1; // the line that character is on
  if (text.charCodeAt(0) === 0xfeff) pos = 1;

  // The record being read: the line it starts on and the index in rows it takes.
  let recordLine = 1;
  let recordRow = null;

  // How the field just read ended: COMMA, LINE_BREAK or END.
  let fEnd = END;

  // The next of `,`, LF or CR at or after pos (one native scan).
  const DELIM = /[,\n\r]/g;

  // Reads unquoted characters onto `value` up to the next comma or line break:
  // a CR not followed by LF is dropped (stray_cr). Sets fEnd. The common case
  // finds the next comma or LF with two native scans and takes one slice, so no
  // match object is ever built; only a field holding a stray CR falls back to
  // the delimiter scan below.
  function readUnquoted(value) {
    const ci = text.indexOf(',', pos);
    const ni = text.indexOf('\n', pos);
    let end;
    let ended;
    if (ci === -1) {
      if (ni === -1) { end = len; ended = END; }
      else { end = ni; ended = LINE_BREAK; }
    } else if (ni === -1) { end = ci; ended = COMMA; }
    else if (ci < ni) { end = ci; ended = COMMA; }
    else { end = ni; ended = LINE_BREAK; }
    // A CR right before the LF is only the CRLF ending, never a stray CR.
    let segEnd = end;
    if (ended === LINE_BREAK && end > pos && text.charCodeAt(end - 1) === 13) segEnd = end - 1;
    const seg = text.slice(pos, segEnd);
    if (seg.indexOf('\r') === -1) {
      value += seg;
      if (ended === END) pos = len;
      else {
        pos = end + 1;
        if (ended === LINE_BREAK) line += 1;
      }
      fEnd = ended;
      return value;
    }
    return readUnquotedSlow(value);
  }

  // The same read, for the rare field holding a stray CR: a CR not followed by
  // LF is dropped (stray_cr) and the field goes on. Sets fEnd.
  function readUnquotedSlow(value) {
    for (;;) {
      DELIM.lastIndex = pos;
      const m = DELIM.exec(text);
      if (m === null) {
        if (pos < len) value += text.slice(pos);
        pos = len;
        fEnd = END;
        return value;
      }
      const d = m.index;
      const c = text.charCodeAt(d);
      if (c === 44) {
        value += text.slice(pos, d);
        pos = d + 1;
        fEnd = COMMA;
        return value;
      }
      if (c === 10) {
        value += text.slice(pos, d);
        pos = d + 1;
        line += 1;
        fEnd = LINE_BREAK;
        return value;
      }
      // A CR: a line break when followed by LF, else a stray CR, dropped.
      if (text.charCodeAt(d + 1) === 10) {
        value += text.slice(pos, d);
        pos = d + 2;
        line += 1;
        fEnd = LINE_BREAK;
        return value;
      }
      errors.push({ code: 'stray_cr', line: recordLine, row: recordRow });
      value += text.slice(pos, d);
      pos = d + 1;
    }
  }

  // The segment text[p..q) as a quoted field holds it: CRLF and lone CR each
  // one LF, and the line count moved past them.
  function quotedSegment(p, q) {
    const seg = text.slice(p, q);
    if (seg.indexOf('\n') === -1 && seg.indexOf('\r') === -1) return seg;
    const breaks = seg.match(/\r\n?|\n/g);
    if (breaks !== null) line += breaks.length;
    return seg.replace(/\r\n?/g, '\n');
  }

  // Reads a quoted field whose opening quote is at pos. Sets fEnd.
  function readQuoted() {
    const open = pos;
    const openLine = line;
    let value = '';
    let p = pos + 1;
    for (;;) {
      const q = text.indexOf('"', p);
      if (q === -1) {
        // The quote never closes: it is a literal character, and the text
        // after it is read again as unquoted.
        line = openLine;
        errors.push({ code: 'unterminated_quote', line: recordLine, row: recordRow });
        pos = open + 1;
        return readUnquoted('"');
      }
      if (text.charCodeAt(q + 1) === 34) {
        // A doubled quote: one quote in the value, and the field goes on.
        value += quotedSegment(p, q);
        value += '"';
        p = q + 2;
        continue;
      }
      value += quotedSegment(p, q);
      pos = q + 1;
      if (pos >= len) {
        fEnd = END;
        return value;
      }
      const c = text.charCodeAt(pos);
      if (c === 44) {
        pos += 1;
        fEnd = COMMA;
        return value;
      }
      if (c === 10) {
        pos += 1;
        line += 1;
        fEnd = LINE_BREAK;
        return value;
      }
      if (c === 13 && text.charCodeAt(pos + 1) === 10) {
        pos += 2;
        line += 1;
        fEnd = LINE_BREAK;
        return value;
      }
      // What follows the closing quote is appended as it stands, up to the
      // next comma or line break.
      const junk = readUnquoted(value);
      if (junk.length > value.length) {
        errors.push({ code: 'quote_junk', line: recordLine, row: recordRow });
      }
      return junk;
    }
  }

  // Takes a record's fields, in the order read, and files them: the header,
  // or a row at the first record's width.
  function fileRecord(fields, quoted) {
    // A line that held only stray CRs is left with no characters: no record.
    if (fields.length === 1 && fields[0] === '' && !quoted) return;
    if (width < 0) width = fields.length;
    if (header && columns === null) {
      columns = fields;
      return;
    }
    if (fields.length < width) {
      errors.push({ code: 'short_row', line: recordLine, row: recordRow, fields: fields.length });
      while (fields.length < width) fields.push('');
    } else if (fields.length > width) {
      errors.push({ code: 'long_row', line: recordLine, row: recordRow, extra: fields.splice(width) });
    }
    rows.push(fields);
  }

  while (true) {
    // Lines with no characters are no records.
    while (pos < len) {
      const c = text.charCodeAt(pos);
      if (c === 10) {
        pos += 1;
        line += 1;
      } else if (c === 13 && text.charCodeAt(pos + 1) === 10) {
        pos += 2;
        line += 1;
      } else {
        break;
      }
    }
    if (pos >= len) break;

    recordLine = line;
    recordRow = header && columns === null ? null : rows.length;

    // The fast path: a record with no quote and no CR inside it (a CR right
    // before the LF is only its CRLF ending) is one slice, split on commas.
    const nl = text.indexOf('\n', pos);
    const end = nl === -1 ? len : nl;
    let contentEnd = end;
    if (nl !== -1 && end > pos && text.charCodeAt(end - 1) === 13) contentEnd = end - 1;
    const rec = text.slice(pos, contentEnd);
    if (rec.indexOf('"') === -1 && rec.indexOf('\r') === -1) {
      fileRecord(rec.split(','), false);
      pos = nl === -1 ? len : nl + 1;
      if (nl !== -1) line += 1;
      continue;
    }

    // The slow path: a record holding a quote or a CR, read a field at a time.
    const fields = [];
    let quoted = false;
    do {
      quoted = text.charCodeAt(pos) === 34;
      fields.push(quoted ? readQuoted() : readUnquoted(''));
    } while (fEnd === COMMA);
    fileRecord(fields, quoted);
  }

  return { columns: header ? (columns ?? []) : null, rows, errors };
}
