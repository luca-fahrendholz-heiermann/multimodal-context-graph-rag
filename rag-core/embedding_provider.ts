export interface EmbeddingProvider {
  embed(texts: string[]): Promise<number[][]>;
}

export class DummyEmbeddingProvider implements EmbeddingProvider {
  async embed(texts: string[]): Promise<number[][]> {
    const dimensions = Number(process.env.EMBEDDING_DIMENSIONS ?? '1536');
    return texts.map((text) => {
      const value = Math.min(1, text.length / 1000);
      return Array.from({ length: dimensions }, (_, idx) => (idx === 0 ? value : 0));
    });
  }
}
