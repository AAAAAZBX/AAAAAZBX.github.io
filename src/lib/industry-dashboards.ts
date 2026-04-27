export type ResourceLink = {
  title: string;
  href: string;
  lang: "中文" | "English" | "中英";
  note: string;
};

export type IndustryDashboard = {
  slug: "quant";
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

export const industryDashboards: Record<"quant", IndustryDashboard> = {
  quant: {
    slug: "quant",
    route: "/quant",
    kicker: "Quant Research Radar",
    title: "金融量化系统搭建全指南",
    subtitle:
      "从零搭建个人量化交易系统：资金规划、知识体系、Python/C++ 开发栈、因子研究、回测框架、实盘部署全流程指南。专为计算机背景的量化开发者设计，含详细学习路径、书单、工具栈和进度跟踪计划。",
    horizon: "2026 核心方向：高频数据工程、C++ 低延迟因子库、机器学习因子挖掘、组合风险优化、LLM 辅助投研、个人量化系统工程化。",
    focus: ["资金与风控", "金融知识", "因子研究", "Python/C++ 开发", "回测系统", "实盘部署"],
    signals: [
      { label: "起点", value: "资金量 1-10万", note: "实盘前先用模拟/小资金" },
      { label: "核心差距", value: "金融知识", note: "计算机背景需补金融基础" },
      { label: "工程优势", value: "C++ 低延迟", note: "你的专业背景可构建高性能系统" },
      { label: "研究关键", value: "可解释因子", note: "避免过拟合，重视 IC/IR" },
    ],
    tracks: [
      {
        title: "📊 一、资金规划与风控底线",
        items: [
          "【1-5 万】：模拟盘 + 极小实盘（单笔 <1000 元），专注学习和验证系统",
          "【5-10 万】：可部署 1-2 个简单策略（如动量、均值回归），每策略单笔 <5000",
          "【10-20 万】：多策略组合，开始考虑仓位管理、止损规则、组合风险",
          "【20 万+】：全系统实盘，考虑融资、多资产、自动化交易",
          "⚠️ 重要：实盘前至少 6 个月模拟测试 + 样本外验证，单策略最大回撤控制在 15% 以内",
        ],
      },
      {
        title: "📚 二、金融知识体系（计算机背景补课清单）",
        items: [
          "✅ 市场基础：股票/期货/期权/债券的区别、交易规则、T+1/T+0、涨跌停、手续费、印花税、滑点",
          "✅ 金融理论：有效市场假说、CAPM、APT、因子模型、现代投资组合理论（马科维茨）、Black-Litterman",
          "✅ 衍生品：期权定价（BS 公式、希腊字母）、期货基差、套利逻辑",
          "✅ 风险管理：VaR、最大回撤、夏普比率、索提诺比率、信息比率、换手率成本",
          "✅ 市场微观结构：订单簿、流动性、买卖价差、冲击成本、高频交易原理",
        ],
      },
      {
        title: "🔬 三、量化研究核心能力",
        items: [
          "因子挖掘：传统因子（动量、反转、价值、质量、成长、波动率、流动性）+ 机器学习因子",
          "因子有效性检验：IC（信息系数）、IR（信息比率）、分组回测、换手率、衰减周期",
          "因子组合：正交化、标准化、加权合成（等权、IC 加权、机器学习组合）",
          "回测陷阱：幸存者偏差、前视偏差、过拟合、交易成本、流动性约束、停牌涨跌停",
          "策略类型：多因子选股、统计套利、配对交易、事件驱动、日内高频、CTA 趋势",
        ],
      },
      {
        title: "💻 四、Python/C++ 开发栈（你的核心优势）",
        items: [
          "Python 研究栈：pandas/numpy（数据处理）、sklearn（机器学习）、statsmodels（统计）、matplotlib/ECharts（可视化）",
          "C++ 高性能栈：Eigen（矩阵计算）、Arrow/Parquet（数据存储）、ZeroMQ（消息通信）、Boost（工具库）",
          "数据工程：Tushare/AkShare（A 股数据）、RiceQuant/JoinQuant（平台）、DolphinDB/ClickHouse（时序数据库）",
          "回测框架：自行实现向量化回测 vs 使用 Backtrader/Zipline（建议先自己写）",
          "系统架构：行情接收 → 因子计算 → 信号生成 → 订单管理 → 风控检查 → 交易执行 → 监控告警",
        ],
      },
      {
        title: "🚀 五、从模拟到实盘的关键路径",
        items: [
          "阶段 1：单因子研究（1-2 个月）→ 输出因子报告（IC、分组收益、换手）",
          "阶段 2：多因子组合 + 回测（2-3 个月）→ 实现完整回测框架，加入交易成本",
          "阶段 3：模拟盘验证（3-6 个月）→ 实时计算因子，模拟下单，跟踪偏差",
          "阶段 4：极小资金实盘（6-12 个月）→ 单策略，严格风控，记录每笔交易",
          "阶段 5：多策略 + 自动化（12 个月+）→ 组合优化、仓位管理、自动化交易",
        ],
      },
    ],
    roadmap: [
      {
        stage: "第 1 个月",
        title: "金融基础 + 工具链搭建",
        tasks: [
          "📖 读书：《漫步华尔街》《打开量化投资的黑箱》《主动投资组合管理》第 1-3 章",
          "💻 工具：安装 Python 3.10+、VSCode/PyCharm、Git、Docker（可选）",
          "📊 数据：注册 Tushare/AkShare，下载沪深 300 成分股日线数据（2015-2023）",
          "🎯 产出：能用 pandas 读取、清洗、复权股票数据，计算基础指标（收益率、波动率）",
        ],
      },
      {
        stage: "第 2-3 个月",
        title: "单因子研究与回测框架",
        tasks: [
          "📖 读书：《量化投资：策略与技术》《因子投资：方法与实践》第 1-4 章",
          "🔬 因子：实现 5 个经典因子（动量 20 日、反转 5 日、PE、ROE、波动率）",
          "📈 回测：写第一个向量化回测（月度调仓、等权组合、手续费 0.03%）",
          "🎯 产出：单因子报告（IC 均值、IR、分组累计收益、最大回撤、换手率）",
        ],
      },
      {
        stage: "第 4-6 个月",
        title: "多因子组合 + C++ 原型",
        tasks: [
          "📖 读书：《因子投资：方法与实践》第 5-9 章、《金融计量学：时间序列分析视角》",
          "🧮 组合：因子标准化、正交化、等权/IC 加权合成",
          "⚡ C++：用 Eigen 实现因子计算核心（如移动平均、标准差），对比 Python 性能",
          "🎯 产出：多因子策略回测报告（年化收益、夏普、最大回撤、月度胜率）",
        ],
      },
      {
        stage: "第 7-9 个月",
        title: "风险模型 + 组合优化",
        tasks: [
          "📖 读书：《主动投资组合管理》第 4-10 章、《Risk and Asset Allocation》",
          "🛡️ 风控：计算组合协方差矩阵、跟踪误差、行业暴露",
          "⚖️ 优化：实现最小方差组合、风险平价、带约束的均值-方差优化（cvxpy）",
          "🎯 产出：不同优化方法对比实验 + 风险归因分析",
        ],
      },
      {
        stage: "第 10-12 个月",
        title: "模拟盘 + 实盘准备",
        tasks: [
          "🤖 模拟：接入实时行情（Tushare Pro/券商 API），每日计算因子和信号",
          "📊 监控：搭建监控看板（净值曲线、持仓、因子暴露、风险指标）",
          "📝 文档：写策略说明书（逻辑、参数、风控规则、历史表现）",
          "🎯 产出：6 个月模拟盘记录 + 实盘交易计划（资金分配、止损规则）",
        ],
      },
      {
        stage: "第 13-18 个月",
        title: "极小资金实盘 + 迭代",
        tasks: [
          "💰 实盘：1-2 个策略，单策略资金 <10% 总资产，手动下单或半自动",
          "📊 复盘：每周复盘交易记录，对比模拟与实盘偏差",
          "🔍 归因：分析收益来源（因子贡献、行业配置、个股选择）",
          "🎯 产出：实盘交易日志 + 季度策略评估报告",
        ],
      },
      {
        stage: "第 19-24 个月",
        title: "系统化与自动化",
        tasks: [
          "⚡ C++ 低延迟：重构核心计算模块（因子、信号）为 C++ 库，Python 调用",
          "📡 数据管道：搭建自动数据更新 + 因子计算 + 信号生成流水线",
          "🤖 自动化：实现自动下单（券商 API）、风控检查、异常告警",
          "🎯 产出：个人量化交易系统（Python 前端 + C++ 核心 + 监控后台）",
        ],
      },
    ],
    projects: [
      {
        title: "因子研究实验室（Python）",
        output: "自动下载数据 → 计算 10+ 因子 → 输出因子报告（IC/IR/分组收益/换手）→ 可视化",
        stack: "pandas, numpy, sklearn, matplotlib/ECharts, Tushare/AkShare",
      },
      {
        title: "向量化回测引擎（Python）",
        output: "支持多因子、月度调仓、手续费/滑点、行业/市值中性、绩效分析",
        stack: "pandas（向量化计算）、numpy、cvxpy（优化）、ECharts（可视化）",
      },
      {
        title: "C++ 因子计算库（高性能核心）",
        output: "用 Eigen 实现移动窗口统计（均值、标准差、相关性）、技术指标、因子正交化",
        stack: "C++17, Eigen, Arrow/Parquet, pybind11（Python 绑定）",
      },
      {
        title: "模拟交易监控看板",
        output: "实时显示持仓、净值、因子暴露、风险指标、交易信号历史",
        stack: "FastAPI（后端）、ECharts（前端）、WebSocket（实时推送）",
      },
      {
        title: "组合优化工具箱",
        output: "比较等权、最小方差、风险平价、Black-Litterman、带约束优化的表现",
        stack: "cvxpy, numpy, pandas, matplotlib",
      },
    ],
    resources: [
      {
        title: "📖 金融基础（必读）",
        href: "https://book.douban.com/subject/26607534/",
        lang: "中文",
        note: "《漫步华尔街》- 了解市场基本规律，破除技术分析迷信",
      },
      {
        title: "📖 量化入门（必读）",
        href: "https://book.douban.com/subject/34843373/",
        lang: "中文",
        note: "《打开量化投资的黑箱》- 量化对冲基金工作流程揭秘",
      },
      {
        title: "📖 因子投资（核心）",
        href: "https://book.douban.com/subject/35187361/",
        lang: "中文",
        note: "《因子投资：方法与实践》- 石川，中文因子圣经，必读",
      },
      {
        title: "📖 组合管理（进阶）",
        href: "https://book.douban.com/subject/4171195/",
        lang: "中文",
        note: "《主动投资组合管理》- Grinold & Kahn，量化组合圣经",
      },
      {
        title: "📖 金融计量（数学）",
        href: "https://book.douban.com/subject/26790563/",
        lang: "中文",
        note: "《金融计量学：时间序列分析视角》- 张成思，时间序列基础",
      },
      {
        title: "🌐 DolphinDB 量化课程",
        href: "https://dolphindb.cn/course",
        lang: "中文",
        note: "实战课程，覆盖数据、因子、回测、组合优化、风控归因",
      },
      {
        title: "🌐 JoinQuant 聚宽",
        href: "https://www.joinquant.com/",
        lang: "中文",
        note: "中文量化平台，可快速验证策略原型，学习社区策略",
      },
      {
        title: "🌐 QuantConnect",
        href: "https://www.quantconnect.com/",
        lang: "English",
        note: "支持 Python/C#，全球市场数据，适合学习和实盘（需付费）",
      },
      {
        title: "🌐 QuantStart",
        href: "https://www.quantstart.com/",
        lang: "English",
        note: "量化交易、回测、统计套利、工程实践文章",
      },
      {
        title: "🌐 WorldQuant 101 Alphas",
        href: "https://arxiv.org/abs/1601.00991",
        lang: "English",
        note: "101 个公式化 Alpha，练习因子思维和实现",
      },
      {
        title: "🌐 QuantConnect Lean Engine",
        href: "https://github.com/QuantConnect/Lean",
        lang: "English",
        note: "开源 C# 量化引擎，可参考架构设计（C++ 可借鉴）",
      },
      {
        title: "🌐 Backtrader",
        href: "https://www.backtrader.com/",
        lang: "English",
        note: "Python 回测框架，可参考事件驱动设计（但建议先自己写）",
      },
      {
        title: "🎓 Open Yale: Financial Markets",
        href: "https://oyc.yale.edu/economics/econ-252-11",
        lang: "English",
        note: "Robert Shiller 金融市场公开课，培养金融直觉",
      },
      {
        title: "🎓 MIT Finance Theory I",
        href: "https://ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008/",
        lang: "English",
        note: "MIT 金融理论，资产定价、投资组合理论基础",
      },
    ],
    watchlist: [
      "因子有效性衰减",
      "交易成本建模",
      "组合风险归因",
      "C++ 低延迟优化",
      "机器学习防过拟合",
      "实盘与模拟偏差",
      "市场风格切换",
      "流动性风险",
      "LLM 辅助投研",
      "另类数据（新闻/舆情）",
    ],
  },
};
