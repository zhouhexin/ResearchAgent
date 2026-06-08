import { useMemo, useState } from "react";
import { Alert, Button, ConfigProvider, Empty, Input, Layout, List, Typography } from "antd";
import zhCN from "antd/locale/zh_CN";
import { askQuestion } from "./api";
import "./styles.css";

const { Header, Content, Sider } = Layout;
const { Text, Title, Paragraph } = Typography;

interface HistoryItem {
  id: string;
  question: string;
  answer: string;
  createdAt: string;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canSubmit = useMemo(() => query.trim().length > 0 && !loading, [query, loading]);

  async function submitQuestion() {
    const trimmed = query.trim();
    if (!trimmed) {
      setError("请输入问题或关键词。");
      return;
    }

    setLoading(true);
    setError("");
    setAnswer("");

    try {
      const result = await askQuestion(trimmed);
      if (result.error) {
        setError(result.error);
        return;
      }
      const finalAnswer = result.answer || "没有生成可展示的回答。";
      setAnswer(finalAnswer);
      setHistory((items) => [
        {
          id: `${Date.now()}`,
          question: trimmed,
          answer: finalAnswer,
          createdAt: formatTime(new Date()),
        },
        ...items,
      ]);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "请求失败，请稍后重试。");
    } finally {
      setLoading(false);
    }
  }

  function clearCurrent() {
    setQuery("");
    setAnswer("");
    setError("");
  }

  return (
    <ConfigProvider locale={zhCN}>
      <Layout className="app-shell">
        <Header className="app-header">
          <div>
            <Title level={3} className="app-title">
              实验室知识问答系统
            </Title>
            <Text className="app-subtitle">基于论文知识库的问答入口</Text>
          </div>
        </Header>
        <Layout className="app-main">
          <Content className="qa-surface">
            <section className="query-panel">
              <Text strong>问题或关键词</Text>
              <Input.TextArea
                className="query-input"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="例如：DepthDark 在哪些数据集上进行了训练？"
                autoSize={{ minRows: 4, maxRows: 8 }}
                onPressEnter={(event) => {
                  if ((event.metaKey || event.ctrlKey) && canSubmit) {
                    submitQuestion();
                  }
                }}
              />
              <div className="actions">
                <Button type="primary" size="large" loading={loading} disabled={!canSubmit} onClick={submitQuestion}>
                  提交问题
                </Button>
                <Button size="large" onClick={clearCurrent} disabled={loading}>
                  清空
                </Button>
              </div>
            </section>

            <section className="answer-panel">
              <div className="answer-heading">
                <Text strong>最终回答</Text>
              </div>
              {error ? <Alert type="error" showIcon message={error} /> : null}
              {!error && !answer && !loading ? (
                <Empty description="提交问题后，这里会显示最终回答。" />
              ) : null}
              {loading ? <Alert type="info" showIcon message="正在生成回答，请稍候。" /> : null}
              {answer ? (
                <Paragraph className="answer-text">
                  {answer}
                </Paragraph>
              ) : null}
            </section>
          </Content>
          <Sider width={320} className="history-sider" breakpoint="lg" collapsedWidth={0}>
            <div className="history-header">
              <Text strong>当前会话历史</Text>
              {history.length ? (
                <Button type="link" onClick={() => setHistory([])}>
                  清除
                </Button>
              ) : null}
            </div>
            {history.length ? (
              <List
                dataSource={history}
                renderItem={(item) => (
                  <List.Item className="history-item">
                    <button
                      className="history-button"
                      onClick={() => {
                        setQuery(item.question);
                        setAnswer(item.answer);
                        setError("");
                      }}
                    >
                      <Text strong className="history-question">
                        {item.question}
                      </Text>
                      <Text type="secondary">{item.createdAt}</Text>
                    </button>
                  </List.Item>
                )}
              />
            ) : (
              <Empty description="暂无历史记录" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            )}
          </Sider>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}
