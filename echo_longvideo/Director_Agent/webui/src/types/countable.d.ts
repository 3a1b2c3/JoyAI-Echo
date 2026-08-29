declare module "countable" {
  export interface CountableResult {
    paragraphs: number;
    sentences: number;
    words: number;
    characters: number;
    all: number;
  }

  export interface CountableOptions {
    hardReturns?: boolean;
    stripTags?: boolean;
    ignore?: string[];
  }

  interface Countable {
    on(
      elements: Element | Element[],
      callback: (result: CountableResult) => void,
      options?: CountableOptions,
    ): Countable;
    off(elements: Element | Element[]): Countable;
    count(
      target: string | Element,
      callback: (result: CountableResult) => void,
      options?: CountableOptions,
    ): Countable;
    enabled(elements: Element | Element[]): boolean;
  }

  const Countable: Countable;
  export default Countable;
}
