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

  private partsToText(parts: any[]): string {
    return parts
      .map((p) => {
        if (typeof p === 'string') return p;
        if (p && typeof p.text === 'string') return p.text;
        return '';
      })
      .join('');
  }

  private contentToMessage(content: any): { role: string; content: string } {
    const role = content.role === 'model' ? 'assistant' : content.role;
    return {
      role,
      content: this.partsToText(content.parts || []),
    };
  }

  private toMessages(req: GenerateContentParameters): any[] {
    const items = req.contents as any;
    if (Array.isArray(items) && items.length && 'role' in items[0]) {
      return items.map((c: any) => this.contentToMessage(c));
    }
    if (Array.isArray(items)) {
      return [{ role: 'user', content: this.partsToText(items) }];
    }
    if ('parts' in items) {
      return [this.contentToMessage(items)];
    }
    return [{ role: 'user', content: this.partsToText([items]) }];
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

  async countTokens(
    _req: CountTokensParameters,
  ): Promise<CountTokensResponse> {
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

