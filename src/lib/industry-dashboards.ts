export type ResourceLink = {
  title: string;
  href: string;
  lang: "中文" | "English" | "中英";
  note: string;
};

export type IndustryDashboard = {
  slug: "ai" | "fintech" | "quant";
  title: string;
  kicker: string;
  subtitle: string;
  route: string;
  horizon: string;
  focus: string[];
  signals: { label: string; value: string; note: string }[];
  tracks: { title: string; items: string[] }[];
  roadmap: { stage: string; title: string; tasks: string[] }[];
  projects: { title: string; output: string; stack: string }[];
  resources: ResourceLink[];
  watchlist: string[];
};

export const industryDashboards: Record<"ai" | "fintech" | "quant", IndustryDashboard> = {
  ai: {
    slug: "ai",
    route: "/ai",
    kicker: "AI Industry Radar",
    title: "AI 大模型算法与工程路线",
    subtitle:
      "面向大模型算法、RAG、Agent、评测、LLMOps 与 AI Infra 的长期追踪页。目标不是追热点，而是形成可面试、可做项目、可读论文的能力栈。",
    horizon: "2026 核心方向：多模态、长上下文、Agent 工程化、RAG 评测、端侧/本地模型、推理加速与数据闭环。",
    focus: ["LLM 基础", "RAG / Agent", "模型微调", "评测体系", "AI Infra", "论文复现"],
    signals: [
      { label: "主线", value: "RAG + Agent", note: "最贴近工程落地与面试项目" },
      { label: "壁垒", value: "Eval", note: "能衡量系统质量的人更稀缺" },
      { label: "底座", value: "PyTorch", note: "算法岗仍绕不开训练与模型结构" },
      { label: "作品集", value: "3-5 个项目", note: "比只刷课程更有效" },
    ],
    tracks: [
      {
        title: "知识体系",
        items: [
          "数学：线性代数、概率统计、优化、信息论基础。",
          "深度学习：反向传播、CNN/RNN/Transformer、归一化、优化器、正则化。",
          "LLM：Tokenizer、位置编码、Attention、预训练、SFT、DPO/RLHF、MoE、长上下文。",
          "应用工程：Prompt、RAG、Agent、工具调用、工作流编排、向量数据库、缓存、权限与审计。",
          "评测：离线测试集、LLM-as-judge、RAGAS、任务成功率、成本/延迟/稳定性指标。",
          "部署：推理服务、量化、vLLM/TensorRT-LLM、监控、灰度、日志与反馈数据闭环。",
        ],
      },
      {
        title: "当前进展追踪",
        items: [
          "模型能力从单轮问答转向可执行任务的 Agentic workflow。",
          "企业落地重点从“接 API”转向数据接入、权限、可观测、评测和成本控制。",
          "开源模型生态持续增强，Qwen、DeepSeek、Llama、Gemma 等适合作为本地实验底座。",
          "多模态、代码 Agent、数据分析 Agent 是高价值应用场景。",
        ],
      },
      {
        title: "能力缺口",
        items: [
          "只会调 API 不够，需要理解模型限制、评测、检索和系统设计。",
          "算法岗需要论文复现与训练经验；工程岗需要稳定系统与产品闭环。",
          "做项目时必须展示失败样例、评测集、改进过程，而不是只展示 demo。",
        ],
      },
    ],
    roadmap: [
      {
        stage: "0-1 月",
        title: "补齐底座",
        tasks: ["PyTorch 手写 MLP/Transformer 小模型", "复习概率统计与优化", "熟悉 Hugging Face Transformers/Datasets"],
      },
      {
        stage: "1-3 月",
        title: "RAG 工程",
        tasks: ["做一个个人知识库 RAG", "实现 chunking、hybrid search、rerank、引用溯源", "建立 50-100 条评测集"],
      },
      {
        stage: "3-5 月",
        title: "Agent 与工具调用",
        tasks: ["做一个网页/论文/数据分析 Agent", "加入工具权限、失败重试、轨迹日志", "统计任务成功率和平均成本"],
      },
      {
        stage: "5-8 月",
        title: "算法/Infra 深化",
        tasks: ["复现 LoRA/DPO/RAGAS 或 FlashAttention 相关内容", "部署本地模型推理服务", "写技术博客沉淀"],
      },
    ],
    projects: [
      {
        title: "个人论文研究 Agent",
        output: "输入论文主题，自动检索、总结、对比方法、生成阅读路线。",
        stack: "OpenAI API / local LLM, vector DB, rerank, eval set",
      },
      {
        title: "博客知识库 RAG",
        output: "把自己的 Markdown 博客变成可问答系统，展示引用来源和置信度。",
        stack: "Astro content, embeddings, Supabase/SQLite, RAGAS-style eval",
      },
      {
        title: "LLM Eval Dashboard",
        output: "比较不同提示词/模型/检索策略在固定测试集上的得分、成本和延迟。",
        stack: "Python, ECharts, JSONL dataset, LLM-as-judge",
      },
    ],
    resources: [
      {
        title: "OpenAI Agent Evals 文档",
        href: "https://platform.openai.com/docs/guides/agent-evals",
        lang: "English",
        note: "学习生产级 Agent 如何做可复现评测。",
      },
      {
        title: "OpenAI AgentKit 发布介绍",
        href: "https://openai.com/index/introducing-agentkit/",
        lang: "English",
        note: "了解 Agent 工作流、Evals 和企业级工具化趋势。",
      },
      {
        title: "OpenAI Data Agent 案例",
        href: "https://openai.com/index/inside-our-in-house-data-agent/",
        lang: "English",
        note: "重点看数据 Agent 的评测方法和 SQL 正确性验证。",
      },
      {
        title: "Hugging Face Course",
        href: "https://huggingface.co/learn",
        lang: "English",
        note: "模型、Transformers、Agents、开源生态入门。",
      },
      {
        title: "李沐《动手学深度学习》",
        href: "https://zh.d2l.ai/",
        lang: "中文",
        note: "中文深度学习基础首选，适合补 PyTorch 和模型结构。",
      },
      {
        title: "Papers with Code",
        href: "https://paperswithcode.com/",
        lang: "English",
        note: "追踪论文、榜单和开源实现。",
      },
    ],
    watchlist: ["RAG 评测", "Agent workflow", "多模态模型", "端侧模型", "推理成本", "数据闭环", "模型安全"],
  },
  fintech: {
    slug: "fintech",
    route: "/fintech",
    kicker: "FinTech Radar",
    title: "金融科技行业路线",
    subtitle:
      "覆盖支付、信贷、风控、财富管理、监管科技、区块链与 AI in Finance。目标是理解金融业务 + 技术系统 + 风险合规三条线。",
    horizon: "2026 核心方向：AI 风控、智能投顾、开放银行/API、反欺诈、监管科技、数据治理、金融大模型。",
    focus: ["支付清算", "信贷风控", "监管科技", "AI in Finance", "开放银行", "金融数据治理"],
    signals: [
      { label: "业务底座", value: "金融产品", note: "先理解钱如何流动和定价" },
      { label: "技术底座", value: "数据 + 风控", note: "FinTech 的核心不是 App，而是风控系统" },
      { label: "合规底线", value: "监管", note: "金融场景不能只讲算法效果" },
      { label: "作品集", value: "风控/反欺诈", note: "最适合学生做可展示项目" },
    ],
    tracks: [
      {
        title: "知识体系",
        items: [
          "金融基础：货币、利率、银行、证券、保险、支付、信贷、资产管理。",
          "技术基础：数据库、后端服务、消息队列、实时计算、数据仓库、风控特征平台。",
          "风控建模：信用评分、反欺诈、异常检测、图模型、模型解释性、拒绝推断。",
          "监管合规：KYC、AML、数据隐私、模型风险管理、审计留痕。",
          "AI in Finance：客服、投研、合规审查、研报摘要、智能投顾、运营自动化。",
        ],
      },
      {
        title: "当前进展追踪",
        items: [
          "金融机构更关注可解释、可审计、可控的 AI，而不是纯黑盒能力。",
          "大模型进入投研、客服、合规、内部知识库，但高风险决策仍需要规则与人工复核。",
          "实时风控、图风控、设备指纹、行为序列建模仍是反欺诈重点。",
          "开放 API、嵌入式金融和支付基础设施继续重塑业务形态。",
        ],
      },
      {
        title: "能力缺口",
        items: [
          "懂机器学习但不懂业务规则，项目很难落地。",
          "懂金融概念但不会工程实现，也难进技术岗。",
          "需要能把模型指标翻译成业务指标：坏账率、召回率、误杀率、资金成本、合规风险。",
        ],
      },
    ],
    roadmap: [
      {
        stage: "0-1 月",
        title: "金融业务地图",
        tasks: ["梳理银行/券商/支付/保险/资管业务", "学习利率、信用、风险、监管基本概念", "读 MIT FinTech 课程"],
      },
      {
        stage: "1-3 月",
        title: "风控建模",
        tasks: ["做一个信用评分项目", "实现 AUC/KS/PSI/稳定性分析", "输出模型解释和风控策略报告"],
      },
      {
        stage: "3-5 月",
        title: "反欺诈系统",
        tasks: ["构造交易流水和用户行为序列", "实现规则引擎 + 异常检测", "做实时告警 dashboard"],
      },
      {
        stage: "5-8 月",
        title: "AI 金融应用",
        tasks: ["做研报/公告 RAG", "做合规问答或客服 Agent", "加入审计日志与人工复核流程"],
      },
    ],
    projects: [
      {
        title: "信用评分卡 + 机器学习风控",
        output: "从数据清洗、特征工程、评分卡、XGBoost 到模型报告。",
        stack: "Python, pandas, sklearn, XGBoost, Streamlit/ECharts",
      },
      {
        title: "交易反欺诈实时看板",
        output: "模拟交易流水，检测异常设备、异常 IP、短时高频交易。",
        stack: "FastAPI, PostgreSQL/Supabase, rules engine, dashboard",
      },
      {
        title: "金融文档 RAG",
        output: "对年报、公告、监管文件做检索问答和引用溯源。",
        stack: "PDF parser, embeddings, vector DB, LLM eval",
      },
    ],
    resources: [
      {
        title: "MIT FinTech: Shaping the Financial World",
        href: "https://ocw.mit.edu/courses/15-s08-fintech-shaping-the-financial-world-spring-2020/",
        lang: "English",
        note: "系统看 FinTech、AI、开放 API、支付、信贷、资本市场。",
      },
      {
        title: "MIT Class 3: Artificial Intelligence in Finance",
        href: "https://www.youtube.com/watch?v=OUAMdi281mQ",
        lang: "English",
        note: "AI in Finance 的经典公开课视频。",
      },
      {
        title: "MIT Consumer Finance and FinTech",
        href: "https://orbit.mit.edu/classes/consumer-finance-and-fintech-15.483",
        lang: "English",
        note: "消费金融、产品设计与金融创新视角。",
      },
      {
        title: "中国人民银行",
        href: "https://www.pbc.gov.cn/",
        lang: "中文",
        note: "政策、支付、金融科技监管相关信息源。",
      },
      {
        title: "国家金融监督管理总局",
        href: "https://www.nfra.gov.cn/",
        lang: "中文",
        note: "银行保险监管、风险治理、政策信息。",
      },
      {
        title: "中国信通院",
        href: "https://www.caict.ac.cn/",
        lang: "中文",
        note: "数字经济、金融科技、AI 治理相关报告值得跟踪。",
      },
    ],
    watchlist: ["金融大模型", "智能风控", "反欺诈", "开放银行", "监管科技", "隐私计算", "支付基础设施"],
  },
  quant: {
    slug: "quant",
    route: "/quant",
    kicker: "Quant Research Radar",
    title: "金融量化研究路线",
    subtitle:
      "面向量化研究、因子挖掘、回测、组合优化、风险管理和交易系统。重点是研究闭环：假设、数据、因子、回测、归因、实盘约束。",
    horizon: "2026 核心方向：另类数据、机器学习因子、组合优化、风险归因、低延迟数据工程、LLM 辅助投研。",
    focus: ["数学统计", "金融市场", "因子研究", "回测系统", "组合优化", "交易工程"],
    signals: [
      { label: "研究核心", value: "因子", note: "收益来源必须可解释、可检验" },
      { label: "工程核心", value: "数据", note: "清洗、对齐、复权、幸存者偏差是基础" },
      { label: "风控核心", value: "回撤", note: "收益不是唯一目标" },
      { label: "作品集", value: "研究报告", note: "比只放代码更像量化研究员" },
    ],
    tracks: [
      {
        title: "知识体系",
        items: [
          "数学：概率统计、时间序列、随机过程、优化、矩阵分析。",
          "金融：股票、期货、期权、债券、市场微观结构、交易成本、风险模型。",
          "因子：动量、反转、价值、质量、成长、波动率、流动性、事件驱动。",
          "回测：数据清洗、复权、调仓、滑点、手续费、成交约束、过拟合控制。",
          "组合：均值方差、风险平价、Black-Litterman、约束优化、归因分析。",
          "工程：行情数据、K 线/快照、特征流水线、任务调度、研究平台、实盘监控。",
        ],
      },
      {
        title: "当前进展追踪",
        items: [
          "机器学习用于非线性因子组合、风险预测、事件分类，但需要严格防过拟合。",
          "量化平台越来越重视数据工程与研究效率，DolphinDB/ClickHouse/Arrow 等高性能数据栈常见。",
          "LLM 更适合辅助研报、公告、新闻、代码生成和研究流程自动化，不应直接替代交易决策。",
          "多资产、多周期和组合层面的风险管理比单策略收益更重要。",
        ],
      },
      {
        title: "能力缺口",
        items: [
          "很多策略 demo 忽略手续费、滑点、停牌、涨跌停和幸存者偏差。",
          "量化面试常考数学统计、Python、数据处理和策略解释。",
          "要能写研究报告：数据来源、假设、分组回测、IC/IR、换手、回撤、归因。",
        ],
      },
    ],
    roadmap: [
      {
        stage: "0-1 月",
        title: "基础与工具",
        tasks: ["复习概率统计/时间序列", "熟练 pandas/numpy/sklearn", "完成股票/基金/期货基础知识"],
      },
      {
        stage: "1-3 月",
        title: "单因子研究",
        tasks: ["做 5 个经典因子", "输出 IC、分组收益、换手、回撤", "写成可复现实验报告"],
      },
      {
        stage: "3-5 月",
        title: "多因子与组合",
        tasks: ["做因子正交化/标准化/合成", "实现组合优化和风险约束", "加入交易成本模型"],
      },
      {
        stage: "5-8 月",
        title: "系统化研究平台",
        tasks: ["搭建数据更新、回测、报告生成流水线", "做参数稳定性和样本外检验", "准备量化研究作品集"],
      },
    ],
    projects: [
      {
        title: "A 股多因子研究框架",
        output: "自动计算因子、IC、分组回测、净值曲线、风险归因。",
        stack: "Python, pandas, sklearn, vectorized backtest, ECharts",
      },
      {
        title: "新闻/公告事件驱动研究",
        output: "用 LLM 抽取事件标签，再做事件前后收益统计。",
        stack: "LLM, text parser, event study, backtest",
      },
      {
        title: "组合优化实验室",
        output: "比较等权、最小方差、风险平价、Black-Litterman 的表现。",
        stack: "cvxpy, numpy, matplotlib/ECharts",
      },
    ],
    resources: [
      {
        title: "DolphinDB 量化金融课程",
        href: "https://dolphindb.cn/course",
        lang: "中文",
        note: "中文量化实践课程，覆盖数据、因子、回测、组合优化、风控归因。",
      },
      {
        title: "QuantStart",
        href: "https://www.quantstart.com/",
        lang: "English",
        note: "量化交易、回测、统计套利、工程实践文章。",
      },
      {
        title: "QuantLib",
        href: "https://www.quantlib.org/",
        lang: "English",
        note: "金融工程与衍生品定价经典开源库。",
      },
      {
        title: "Open Yale: Financial Markets",
        href: "https://oyc.yale.edu/economics/econ-252-11",
        lang: "English",
        note: "Robert Shiller 金融市场公开课，适合补金融直觉。",
      },
      {
        title: "JoinQuant 聚宽",
        href: "https://www.joinquant.com/",
        lang: "中文",
        note: "中文量化平台，可用于策略原型和社区学习。",
      },
      {
        title: "WorldQuant 101 Alphas",
        href: "https://arxiv.org/abs/1601.00991",
        lang: "English",
        note: "经典公式化 Alpha 参考，适合练习因子思维。",
      },
    ],
    watchlist: ["多因子", "机器学习因子", "高频数据", "组合优化", "风险归因", "事件驱动", "LLM 投研"],
  },
};
