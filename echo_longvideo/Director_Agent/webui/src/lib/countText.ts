import Countable from "countable";

/** CJK unified ideographs, extensions, kana, and hangul syllables. */
const CJK_RE =
  /[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/;

function countLatinWords(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  let words = 0;
  Countable.count(trimmed, (result) => {
    words = result.words;
  });
  return words;
}

/**
 * Mixed unit count: CJK characters count individually; Latin segments count
 * as words via Countable (e.g. "the" → 1, "你好" → 2).
 */
export function countMixedUnits(text: string): number {
  if (!text) return 0;

  let total = 0;
  let latinBuffer = "";

  for (const char of text) {
    if (CJK_RE.test(char)) {
      if (latinBuffer.trim()) {
        total += countLatinWords(latinBuffer);
        latinBuffer = "";
      }
      total += 1;
    } else if (/\s/.test(char)) {
      if (latinBuffer.trim()) {
        total += countLatinWords(latinBuffer);
        latinBuffer = "";
      }
    } else {
      latinBuffer += char;
    }
  }

  if (latinBuffer.trim()) {
    total += countLatinWords(latinBuffer);
  }

  return total;
}

/** Truncate text so mixed unit count does not exceed `max`. */
export function truncateToMaxUnits(text: string, max: number): string {
  if (max <= 0) return "";
  if (countMixedUnits(text) <= max) return text;

  let result = "";
  for (let i = 1; i <= text.length; i++) {
    const slice = text.slice(0, i);
    if (countMixedUnits(slice) > max) break;
    result = slice;
  }
  return result;
}
