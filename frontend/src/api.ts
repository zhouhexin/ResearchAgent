export interface AskResponse {
  answer: string;
  run_id?: string | null;
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
  let lastError: unknown;

  for (const apiBaseUrl of getApiBaseUrls()) {
    try {
      const response = await fetch(`${apiBaseUrl}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        lastError = new Error(`请求失败：${response.status}`);
        continue;
      }

      return (await response.json()) as AskResponse;
    } catch (error) {
      lastError = error;
    }
  }

  if (lastError instanceof Error) {
    throw lastError;
  }

  throw new Error("请求失败，请检查后端服务是否已启动。");
}
