```markdown
# LLM Memory Benchmark (template)

这是一个轻量的可运行基准模板，用于评测 LLM memory 管道（向量存储 + 检索 + LLM 生成）的性能与质量。设计目标是：可复现、易扩展、能运行典型记忆场景（保留、回忆、更新、遗忘、干扰）。

主要特性
- 使用向量数据库（Chroma / FAISS）存储“记忆片段”。
- 使用 OpenAI Chat API 做检索增强生成（RAG）。
- 支持逐步（incremental）插入记忆并在指定时间点查询（模拟长期记忆与短期记忆交互）。
- 输出可机器读的 results.json（包含每条测试的指标）。

快速开始
1. 准备环境
   - Python 3.10+
   - 创建并激活虚拟环境

2. 安装依赖
   pip install -r requirements.txt
   pip install git+https://github.com/openai/evals.git@main

3. 配置环境变量（在根目录创建 .env 或导出环境变量）
   - OPENAI_API_KEY: 你的 OpenAI API Key
   - CHROMA_DIR (可选): Chroma 数据目录

4. 运行示例（自动化基准）
   python benchmark/runner.py --vectorstore chroma --tests tests/memory_tests.json --output results.json

使用 OpenAI Evals 进行人工/自动评估（可选）
- 安装 evals： pip install git+https://github.com/openai/evals.git@main
- 将本项目的 results.json 作为 evals 的输入或扩展一个自定义 eval（参见 evals 文档）。
- 本仓库包含 scripts/run_evals.sh 为本地运行 evals 的示例流程（需要根据 evals 版本调整）。

文件说明
- benchmark/runner.py         主入口
- benchmark/memory_benchmark.py  基准核心逻辑：按事件流插入记忆并运行查询
- benchmark/vector_store.py   向量库抽象（Chroma / FAISS stub）
- tests/memory_tests.json     示例测试集（多个场景）
- requirements.txt            Python 依赖
- .env.example                环境变量示例
- scripts/run_evals.sh       OpenAI Evals 本地运行示例

评估指标（示例实现）
- exact_match: 查询答案是否与期望答案文本完全匹配（便于自动化）
- contains: 生成文本是否包含期望关键片段
- retrieved_topk_recall: 在 top-k 检索结果里是否包含 ground-truth memory id（如果提供）

扩展建议
- 用 openai-evals 将人工评审纳入 pipeline（可打分 hallucination / consistency）
- 支持更多向量后端（Milvus、Pinecone、Weaviate）
- 加入 latency / throughput 测量模块
- 支持多 embedding model 对比（text-embedding-3-small vs text-embedding-3-large)
```