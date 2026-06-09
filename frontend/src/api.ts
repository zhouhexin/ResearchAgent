export interface AskResponse {
  answer: string;
  run_id?: string | null;
  error?: string | null;
}

export interface FeedbackPayload {
  run_id?: string | null;
  query: string;
  answer: string;
  rating: "accurate" | "inaccurate";
}

export interface FeedbackResponse {
  ok: boolean;
  error?: string | null;
}

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
const fallbackApiBaseUrl = "http://127.0.0.1:8001";

function getApiBaseUrls(): string[] {
  const candidates = [configuredApiBaseUrl, fallbackApiBaseUrl]
    .filter((url): url is string => Boolean(url))
    .map((url) => url.replace(/\/$/, ""));

  return Array.from(new Set(candidates));
}

export async function askQuestion(query: string): Promise<AskResponse> {
  return postJson<AskResponse>("/ask", { query });
}

export async function submitFeedback(payload: FeedbackPayload): Promise<FeedbackResponse> {
  return postJson<FeedbackResponse>("/feedback", payload);
}

async function postJson<TResponse>(path: string, payload: unknown): Promise<TResponse> {
  let lastError: unknown;

  for (const apiBaseUrl of getApiBaseUrls()) {
    try {
      const response = await fetch(`${apiBaseUrl}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        lastError = new Error(`请求失败：${response.status}`);
        continue;
      }

      return (await response.json()) as TResponse;
    } catch (error) {
      lastError = error;
    }
  }

  if (lastError instanceof Error) {
    throw lastError;
  }

  throw new Error("请求失败，请检查后端服务是否已启动。");
}
