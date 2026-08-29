const MIN_CONTENT_LENGTH = 30;

export function extractScriptContent(text: string): string | null {
  const trimmed = text.trim();
  if (trimmed.length < MIN_CONTENT_LENGTH) return null;

  if (isMetaContent(trimmed)) return null;

  const paragraphs = trimmed.split(/\n\n+/).filter((p) => p.trim().length > 0);
  if (paragraphs.length === 0) return null;

  const contentParagraphs = paragraphs.filter((p) => {
    const t = p.trim();
    if (t.length < 10) return false;
    if (/^(好的|没问题|当然|了解|明白|收到)/.test(t) && t.length < 30)
      return false;
    if (isMetaContent(t)) return false;
    return true;
  });

  if (contentParagraphs.length === 0) return null;
  return contentParagraphs.join("\n\n");
}

function isMetaContent(text: string): boolean {
  const metaPatterns = [
    /步骤|流程|规划|计划/,
    /首先.*然后.*最后/s,
    /第[一二三四五六七八九十\d]+步/,
    /我[会将来].*(?:帮你|为你|给你)/,
    /接下来我/,
    /多步骤|分步/,
    /(?:^|\n)\s*\d+[.、)）]\s*/,
  ];
  const matchCount = metaPatterns.filter((p) => p.test(text)).length;
  return matchCount >= 2;
}
