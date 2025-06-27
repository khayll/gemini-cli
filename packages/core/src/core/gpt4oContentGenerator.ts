/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  ContentGenerator,
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Content,
  PartUnion,
} from '@google/genai';

/** Options for the GPT-4o content generator. */
export interface GPT4oContentGeneratorOptions {
  /** Endpoint URL for the GPT-4o API. */
  url: string;
  /** Name of the authentication header (e.g. "Authorization"). */
  authHeaderName: string;
  /** Token to be used for authentication. */
  authToken: string;
}

/**
 * A simple ContentGenerator implementation that proxies requests to a GPT-4o
 * compatible endpoint. Input messages are converted to OpenAI's chat format and
 * the response is mapped back to a {@link GenerateContentResponse} structure.
 */
export class GPT4oContentGenerator implements ContentGenerator {
  constructor(private readonly opts: GPT4oContentGeneratorOptions) {}

  private partsToText(parts: PartUnion[]): string {
    return parts
      .map((p) => {
        if (typeof p === 'string') return p;
        if (p && typeof (p as { text?: string }).text === 'string') {
          return (p as { text?: string }).text as string;
        }
        return '';
      })
      .join('');
  }

  private contentToMessage(content: Content): {
    role: string;
    content: string;
  } {
    const role =
      content.role === 'model' ? 'assistant' : (content.role ?? 'user');
    return {
      role,
      content: this.partsToText((content.parts ?? []) as PartUnion[]),
    };
  }

  private toMessages(
    req: GenerateContentParameters,
  ): Array<{ role: string; content: string }> {
    const items = req.contents as
      | Content
      | Content[]
      | PartUnion
      | PartUnion[]
      | string
      | string[];

    if (Array.isArray(items) && items.length > 0) {
      if (typeof (items[0] as Content).role !== 'undefined') {
        return (items as Content[]).map((c) => this.contentToMessage(c));
      }
      return [
        { role: 'user', content: this.partsToText(items as PartUnion[]) },
      ];
    }

    if ((items as Content).parts) {
      return [this.contentToMessage(items as Content)];
    }

    return [{ role: 'user', content: this.partsToText([items as PartUnion]) }];
  }

  async generateContent(
    req: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = this.toMessages(req);
    const body = {
      model: req.model,
      messages,
      temperature: req.config?.temperature,
      top_p: req.config?.topP,
      max_tokens: req.config?.maxOutputTokens,
      stream: false,
    };
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      [this.opts.authHeaderName]: this.opts.authToken,
    };
    const response = await fetch(this.opts.url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: req.config?.abortSignal,
    });
    const json = await response.json();
    const text = json.choices?.[0]?.message?.content ?? '';
    const candidate = {
      content: { role: 'model', parts: [{ text }] },
    };
    return { candidates: [candidate] } as GenerateContentResponse;
  }

  async generateContentStream(
    req: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const resp = await this.generateContent(req);
    return (async function* () {
      yield resp;
    })();
  }

  async countTokens(_req: CountTokensParameters): Promise<CountTokensResponse> {
    // Token counting is not supported; return undefined.
    return {} as CountTokensResponse;
  }

  async embedContent(
    _req: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Embeddings are not supported.
    return {} as EmbedContentResponse;
  }
}
