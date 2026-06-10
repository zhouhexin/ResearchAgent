# 前端问答系统服务器部署流程

本文档用于把 ResearchAgent 的前端问答系统部署到服务器。推荐结构如下：

```text
Nginx 对外提供前端静态页面
Nginx 将 /ask、/feedback、/papers/file、/health 转发到 FastAPI
FastAPI 后端只监听 127.0.0.1:8001
```

这样前端页面和 API 保持同源访问，部署后不需要暴露后端端口。

## 1. 拉取代码

如果服务器上还没有项目：

```bash
git clone https://github.com/zhouhexin/ResearchAgent.git
cd ResearchAgent
```

如果服务器上已有项目：

```bash
cd /path/to/ResearchAgent
git pull origin main
```

## 2. 准备 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`，至少确认以下配置：

```text
MINIMAX_API_KEY=你的key
MINIMAX_BASE_URL=https://api.minimax.io/v1
MINIMAX_MODEL=MiniMax-M2.7

QA_API_PROMPT_MODE=public_qa
QA_API_TEMPERATURE=0.0
QA_API_RUNS_DIR=experiments/web_runs
```

说明：

- `QA_API_PROMPT_MODE=public_qa`：前端问答使用面向用户的回答提示词。
- `QA_API_TEMPERATURE=0.0`：降低同问不同答的随机性。
- `QA_API_RUNS_DIR=experiments/web_runs`：前端问答 run JSON 单独保存，不和实验 run 混在一起。

## 3. 准备索引

如果服务器上已经有以下文件，可以跳过本步骤：

```text
storage/index.faiss
storage/metadata.json
```

否则执行：

```bash
python app.py index --docs ./data
```

该命令会基于 `data/` 下的文档构建默认 chunk-level FAISS 索引。

## 4. 手动启动后端验证

先用命令行启动一次 FastAPI：

```bash
source .venv/bin/activate
python -m uvicorn api.server:app --host 127.0.0.1 --port 8001
```

另开一个终端测试：

```bash
curl http://127.0.0.1:8001/health
```

正常返回：

```json
{"status":"ok"}
```

验证通过后，可以停止这个手动启动的进程，后续交给 `systemd` 管理。

## 5. 构建前端

进入前端目录：

```bash
cd frontend
npm ci
```

如果使用 Nginx 反向代理 API，构建时可以把 API 地址设为服务器根地址：

```bash
VITE_API_BASE_URL=http://YOUR_SERVER_IP npm run build
```

如果已经绑定域名：

```bash
VITE_API_BASE_URL=https://YOUR_DOMAIN npm run build
```

构建产物会生成在：

```text
frontend/dist/
```

## 6. 配置 Nginx

安装 Nginx 后，新增站点配置。下面示例使用服务器 IP；如果有域名，把 `server_name` 改成域名。

```nginx
server {
    listen 80;
    server_name YOUR_SERVER_IP;

    root /path/to/ResearchAgent/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /ask {
        proxy_pass http://127.0.0.1:8001/ask;
    }

    location /feedback {
        proxy_pass http://127.0.0.1:8001/feedback;
    }

    location /papers/file/ {
        proxy_pass http://127.0.0.1:8001/papers/file/;
    }

    location /health {
        proxy_pass http://127.0.0.1:8001/health;
    }
}
```

检查配置并重载：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 7. 配置 systemd 后端服务

创建服务文件：

```bash
sudo nano /etc/systemd/system/researchagent-api.service
```

示例内容：

```ini
[Unit]
Description=ResearchAgent FastAPI
After=network.target

[Service]
WorkingDirectory=/path/to/ResearchAgent
Environment="PATH=/path/to/ResearchAgent/.venv/bin"
ExecStart=/path/to/ResearchAgent/.venv/bin/python -m uvicorn api.server:app --host 127.0.0.1 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable researchagent-api
sudo systemctl start researchagent-api
```

查看状态：

```bash
sudo systemctl status researchagent-api
```

查看日志：

```bash
sudo journalctl -u researchagent-api -f
```

## 8. 验证部署

浏览器访问：

```text
http://YOUR_SERVER_IP
```

确认以下功能：

- 可以提交问题并得到最终回答。
- 如果回答中提到本地已有论文，回答下方会展示“预览 / 下载”链接。
- 回答中不会展示 run id、chunk id、检索细节。
- 可以点击“准确 / 不准确”提交反馈。
- 点击“准确”的回答会进入当前会话历史。
- 点击“不准确”的回答不会保留在当前会话历史。

服务器上应产生：

```text
experiments/web_runs/             # 前端问答 run JSON
experiments/web_feedback.jsonl    # 用户准确性反馈
experiments/web_paper_index.json  # 本地 PDF 内容标题索引缓存
```

实验运行文件仍然位于：

```text
experiments/runs/
```

## 9. 更新部署

后续更新代码时：

```bash
cd /path/to/ResearchAgent
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
```

如果后端代码或 prompt 有变化：

```bash
sudo systemctl restart researchagent-api
```

如果前端代码有变化：

```bash
cd frontend
npm ci
VITE_API_BASE_URL=http://YOUR_SERVER_IP npm run build
sudo systemctl reload nginx
```

## 10. 常见问题

### 前端页面能打开，但提交问题失败

检查后端服务：

```bash
sudo systemctl status researchagent-api
curl http://127.0.0.1:8001/health
```

检查 Nginx 是否正确代理 `/ask`：

```bash
curl http://YOUR_SERVER_IP/health
```

### 论文预览或下载链接打不开

确认 Nginx 已代理 `/papers/file/`：

```nginx
location /papers/file/ {
    proxy_pass http://127.0.0.1:8001/papers/file/;
}
```

然后检查后端日志：

```bash
sudo journalctl -u researchagent-api -f
```

如果 `experiments/web_paper_index.json` 不存在，重启后端会重新生成：

```bash
sudo systemctl restart researchagent-api
```

### 回答仍然带旧格式

后端 prompt 或答案清洗逻辑更新后，需要重启后端：

```bash
sudo systemctl restart researchagent-api
```

浏览器也需要刷新页面。

### 找不到索引

确认以下文件存在：

```text
storage/index.faiss
storage/metadata.json
```

如果不存在，重新构建：

```bash
python app.py index --docs ./data
```

### 用户反馈没有保存

确认后端进程对项目目录有写权限，并检查：

```text
experiments/web_feedback.jsonl
```

如果文件不存在，先提交一次反馈，再查看后端日志：

```bash
sudo journalctl -u researchagent-api -f
```
