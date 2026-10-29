# StructSynth

StructSynth 是一个面向数学题的结构归纳与同类题生成实验系统。它不把“同类”直接等同于主题或句式相似，而是优先对齐思维步骤、认知操作、依赖关系和主要瓶颈；表面场景、数字、术语表达允许灵活变化。

仓库同时包含：

- 可运行的 Python 谱结构生成流水线；
- 本地中文 Web 实验台；
- 独立的 HyperFrames 无旁白推广片工程；
- 单元测试与 GitHub Actions CI。

## Repository Layout

```text
.
├── src/structsynth/           # Pipeline, spectral method, LLM adapters, Web server
├── tests/                     # Unit tests
├── examples/demo_seeds.json   # Seed questions used by the offline demo
├── promo-structsynth/         # HyperFrames 45s vertical promo project
├── pyproject.toml
└── .github/workflows/ci.yml
```

## 当前管线

1. LLM 把每道种子题切成有序语义语段，并标注结构角色、认知操作和抽象功能。
2. LLM 归纳种子题的思维难度画像：深度、操作集合、依赖链、瓶颈和验证负担。
3. LLM embedding API 编码语段；`M` 同时使用语义相似度和软角色结构相似度，并保留局部邻域连接。
4. 用保持语段连接强度的零模型构造 `N`，计算
   `A = 2 sym(M) - sym(N)`。
5. 对 `A` 做对称特征分解，从领先正谱的特征向量筛高激活语段并聚类，得到 `V'`。
6. LLM 把 `V'` 归纳成软模板槽位，并把思维难度画像注入生成条件。
7. 同一生成模型、相近采样参数下，比较：
   - `baseline`：只给种子题直接仿写；
   - `matrix`：给种子题和只由 `M` 聚类得到的无序语义单元；
   - `spectral`：给种子题和 `V'` 归纳的有序谱模板。
8. 盲评式 LLM judge 输出结构族命中、连续结构相似、认知操作覆盖和思维深度对齐；另计算生成多样性。

## 为什么要屏蔽同题连接

如果 `M` 保留同一道题内部的强相似连接，聚类很容易按“函数题/几何题”之类题材聚集，而不是找到跨题稳定出现的条件、对象、约束、目标等结构角色。默认把同题语段对置零；该选项是 `mask_same_question`。

`N` 当前采用加权配置模型零假设：

```text
N_ij = degree_i * degree_j / sum(degree)
```

这保留了各语段的整体连接强度，但没有特定的社区结构。`sym()` 是矩阵对称化。`A` 的对角线置零。若接入方已有业务定义的 `N`，应替换 `build_contrast()`，不要在论文或实验报告中混称不同零模型。

## 思维难度优先与柔性相似

两个语段可以在具体对象、数字和表达方式上很不同，只要承担相近的结构角色和认知操作，仍然可以在 `M` 中建立连接。默认权重是 68% 语义相似度 + 32% 角色/操作结构相似度。

`neighbor_fraction` 控制每个语段保留多少比例的跨题邻居，默认 35%。它替代单一余弦阈值：

- 连接比例低：结构要求更严格，聚类更干净；
- 连接比例高：更容忍改写、迁移和多种表达方式，但可能混入弱相关语段。

模板槽位现在是软提示。生成器优先保持 `dependency_chain`、`depth_invariants` 和 `bottleneck`，不要求逐字填充槽位，也不要求沿用同一题材。

## 离线试跑

先安装依赖：

```bash
UV_CACHE_DIR="$PWD/work/uv-cache" uv sync --extra dev
source .venv/bin/activate
```

## 启动 Web 界面

```bash
structsynth web --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。界面支持：

- 逐行粘贴题目，或导入 `{"questions": [...]}` 格式的 JSON；
- mock 离线运行与 OpenAI 兼容 API 运行；
- 调整特征向量数、聚类数、高激活分位点、语段邻域比例、生成数量和 temperature；
- 后台执行、阶段进度和运行日志；
- 查看结构角色、思维难度画像、谱诊断、候选语义单元、两套软模板、三组生成题与逐题评测；
- 下载 Markdown 报告或包含全部 JSON 结果的压缩包。

API Key 只随任务提交保存在本机进程内存中，不会写入日志或结果文件。Web 服务默认只绑定回环地址 `127.0.0.1`，没有登录系统，不应直接暴露到公网。

界面端到端冒烟测试可以点击“离线演示”并保留默认示例题。mock 只验证界面和管线是否工作，不用于判断算法效果。

运行确定性离线演示：

```bash
structsynth run \
  --provider mock \
  --seeds examples/demo_seeds.json \
  --out outputs/demo \
  --count 6
```

mock 只用于验收数据流，生成文本和指标不能作为质量结论。

## 接入真实 LLM

当前客户端兼容 OpenAI 风格的 `/chat/completions` 与 `/embeddings`，并允许两者使用不同服务。例如让 DeepSeek 负责生成，另一个服务负责 embedding：

```bash
# Chat：DeepSeek
export QS_LLM_BASE_URL="https://api.deepseek.com"
export QS_LLM_API_KEY="DeepSeek API Key"
export QS_CHAT_MODEL="deepseek-flash"

# Embedding：支持 /embeddings 的服务
export QS_EMBEDDING_BASE_URL="https://api.openai.com/v1"
export QS_EMBEDDING_API_KEY="Embedding API Key"
export QS_EMBEDDING_MODEL="text-embedding-3-small"

structsynth run \
  --provider openai \
  --seeds examples/demo_seeds.json \
  --out outputs/real-run \
  --count 12
```

若不设置 `QS_EMBEDDING_BASE_URL` 或 `QS_EMBEDDING_API_KEY`，会回退到聊天接口的 Base URL/Key，以兼容同时提供聊天和 embedding 的服务。Web 界面中可以直接填写两套配置，也可以点击“DeepSeek 预设”填入官方 Base URL 和聊天模型。

每次运行的 API 调用是批处理的：一次切分、一次思维画像归纳、两次模板归纳、三次生成、按每 20 题分块的评审，另加 embedding。若供应商不支持 `response_format=json_object`，客户端遇到 HTTP 400 会自动重试一次不带该字段的请求。真实供应商如果使用其他鉴权或响应格式，只需实现 `llm.py` 中的 `LLM` 协议。

## 输出

- `segments.json`：语段及原题位置。
- `spectral.json`：矩阵诊断、各特征向量激活最高的语段、候选语义单元。
- `templates.json`：谱模板与仅 `M` 基线的模板。
- `generations.json`：三组生成题及槽位填充。
- `evaluation.json`：逐题评审和分组指标。
- `report.md`：简表与解释边界。

## 实验前必须补上的控制

当前版本是工程 MVP，不是可以直接支持“本算法提升同类题质量”因果结论的实验。正式实验至少要补：

1. **先定义同构。** 不要只用诱导出的模板评自己。应有人工定义的结构标注，例如槽位类型、依赖关系、目标类型和必须保持的约束。
2. **划分训练/留出集。** 模板只能从训练种子归纳；测试种子和人工标准不能泄漏给生成器。
3. **独立、盲评的 judge。** 实验组信息不可进入 judge prompt，并抽样人工复核 judge 的假阳性与假阴性。
4. **公平预算。** 三组使用同一生成模型、temperature、最大 token 和输出数量；prompt token 数应记录。若条件提示明显更长，需要做等 token 或长度敏感性分析。
5. **多样性不能只看题面。** 至少同时报告语义多样性、结构模式重复率、答案/解法重复率和无效题率。
6. **固定 N 的定义。** 现在的 `N` 是可解释的默认零模型，但没有理由把它当成唯一真值；应做随机置换零模型或跨题材负集敏感性分析。
7. **分离“同类”和“同难度”。** 结构族命中回答“是否同类”，思维深度对齐回答“是否同样深”。两者应同时报告并分别做人工标注。

## 已知限制

- 同一题内与同一题型内的语段数量和颗粒度不一致时，会污染矩阵；需要检查切分稳定性。
- K-means 的聚类数会决定 `V'` 大小。建议用特征间隙、轮廓系数和人工可解释性共同选 `k`。
- 特征向量符号任意，比较时只应使用绝对值、相对位置和聚类关系。
- 真实 embedding 的维度、归一化和 API 截断行为会显著影响矩阵；需记录 embedding 模型与版本。

## 推广片工程

`promo-structsynth/` 是独立的 HyperFrames 项目，规格为 45 秒、1080×1920、30fps、无旁白。它包含六幕结构：

1. 换掉题面，保留思维；
2. 函数题从 `2^x` 到 `3^x` 的结构迁移；
3. 语段角色与谱结构；
4. CMO 不等式难度峰值；
5. 产品界面与思维深度对齐；
6. 品牌收束。

本地预览：

```bash
cd promo-structsynth
npm install
npm run dev
```

静态检查与渲染：

```bash
cd promo-structsynth
npx hyperframes@0.8.210 lint
npx hyperframes@0.8.210 render --quality delivery --output renders/structsynth-promo.mp4
```

该推广片不包含 TTS 旁白。音效许可与第三方资源说明见 `THIRD_PARTY_NOTICES.md`。

## License

仓库所有者尚未选择开源许可证。在发布到 GitHub 前，请根据期望的使用范围补充
`LICENSE` 文件。
