---
title: "深度学习软件测试前沿研究综述"
date: "2025-12-14"
description: "综述五篇深度学习&软件测试领域最新研究成果"
tags: ["模糊测试", "测试", "深度学习"]

---

# 深度学习软件测试前沿研究综述

## 总体概览

近年来，随着深度学习在各行业的广泛应用，底层深度学习框架和库（如TensorFlow、PyTorch等）的可靠性愈发重要。传统研究多关注**模型本身**的测试（例如对抗样本测试模型鲁棒性），但越来越多工作开始转向**深度学习库/引擎**的测试，以发现框架实现中的缺陷。本文综述五篇该领域的最新研究工作，分别是 *NeuRI*, *LEMON*, *GraphFuzz*, *FreeFuzz*, *EAGLE*，它们围绕**自动生成测试用例（模型或API调用）\**来揭露深度学习库中的bug。这些工作共享相似的目标，即\**在无需人工干预下高效产生多样化的神经网络测试用例，利用不同形式的差分测试或覆盖引导来发现库实现中的错误**。它们的共同点在于：都自动构造DNN模型（或API序列）作为测试输入，并借助一定的判定机制（如跨库结果不一致、等价变换结果不一致、运行崩溃等）作为测试*oracle*判断是否存在bug。同时，不同方法在**测试对象、测试技术和覆盖引导策略**上各有侧重：

- **NeuRI**：面向*深度学习编译器/库*，通过*归纳规则推理*自动学习算子间约束，从而生成涵盖上百种算子的*多样化有效*模型，极大扩展了测试模型的操作符多样性。它采用从真实轨迹提取约束+符号/具体混合生成的策略，提高了TensorFlow和PyTorch的大量分支覆盖，并发现了大量新缺陷。
- **LEMON**：以*多个深度学习库*为测试对象，提出*库级*的*模型变异测试*方法。LEMON从少量种子模型出发，定义了一系列*模型层级的变异规则*（添加/删除/替换层，改变神经元权重等）生成新模型。然后利用*Keras*作为统一接口将模型分别运行在不同底层库上，对比输出检测不一致，以差分测试找出各库实现差异导致的bug。
- **GraphFuzz**：针对*深度学习推理引擎*，提出*基于计算图*的模糊测试方法。GraphFuzz将模型看作有向图，通过多种*图结构变异*（添加/删除层节点或连接等）生成不同模型；同时引入了*算子级覆盖率*作为反馈指标，并利用*MCTS*（蒙特卡洛树搜索）智能地探索模型空间，从而提升覆盖更多算子实现和执行路径。
- **FreeFuzz**：面向*深度学习库API*本身，提出*API级*的模糊测试方法。不同于以上基于模型结构变异的方法，FreeFuzz通过*挖掘开源代码*（官方文档示例、库自带单元测试、GitHub公开模型等）**收集真实API调用用例**，记录函数参数的动态类型和值空间。然后针对每个API执行定向的参数变异 fuzz，结合**后端差分**（如TensorFlow与其XLA编译后端对比）和**形同测试**（如对函数添加等效变换）来判定是否存在bug。这一方法显著扩大了可测试API覆盖面。
- **EAGLE**：为了解决传统差分测试需多个库实现的局限，EAGLE提出使用*等价计算图*在**单个深度学习库内部**进行差分测试。它预先人工设计并自动化生成了一系列*等价转换规则*（共16条），能够将某个功能用两种不同的计算图表示。对同一库，用等价的两种计算图处理同一输入，若输出不一致则揭示了库实现中的缺陷。这种方法本质是一种*变体的差分测试（类同于元模型测试）*，无需跨库比对即可发现单库中的逻辑错误。

综上，这五项工作从不同角度丰富了深度学习库的测试手段：有的侧重**生成多样模型**（NeuRI、GraphFuzz、LEMON），有的着眼于**挖掘真实调用**（FreeFuzz），有的创新于**等价变换**（EAGLE）。它们的测试对象涵盖了**底层库的不同层面**：从单个API函数到整个模型，从单一框架内部到多框架对比。不少工作已实质性发现并上报了大量库bug（如NeuRI发现100个、FreeFuzz发现49个、EAGLE发现25个等），推动了深度学习框架的健壮性改进。下面我们按从基础到进阶的顺序，对每篇论文进行详细介绍。

## LEMON：库级覆盖引导的深度学习系统模糊测试

**LEMON**（发表在ESEC/FSE 2020）是较早探索*深度学习库测试*的工作之一。它关注的是**底层深度学习框架实现之间的一致性**：不同框架（如TensorFlow、Theano、MXNet、CNTK）在实现相同神经网络时，若产生不同行为或结果，往往说明其中某个存在bug。LEMON的核心思想是：**通过模型级变异生成多种神经网络模型，并利用跨库差分测试来揭露实现差异**。图展示了LEMON的方法流程：

*LEMON 方法概览：从种子模型出发，应用一系列变异生成新模型，用*Keras*将模型运行在多个后端库（TensorFlow、Theano、CNTK、MXNet），比较输出找出不一致。变异选择由反馈（不一致程度）引导，以逐步放大不同库间的输出差异。*

具体而言，LEMON包含两大技术步骤：

1. **模型变异规则**：LEMON设计了丰富的*模型层级变异*操作，包括“整体层操作”（如移除一层、交换两层、复制或插入新层、更改激活函数等）以及“层内部操作”（如对某层部分神经元添加噪声、屏蔽/删除神经元连接等）。例如，“Layer Addition (LA)”规则会从一个通用层池中选取某个层（要求其输入输出形状一致），插入到模型中合适位置，从而引入新的算子调用；“Neuron Effect Block (NEB)”则随机选择30%的神经元，将它们与下一层的连接权重设为0，以削弱该层对后续的影响。通过这些变异，LEMON可以从一个初始种子模型衍生出大量形态各异的模型。此外，LEMON支持**高阶变异**，即多次迭代应用不同规则，产生更复杂的模型改变。
2. **变异决策与差分检测**：由于可能的变异组合非常庞大，LEMON采用了**启发式指导**策略来优先探索那些能“放大库间不一致程度”的变异。具体做法是：首先选择一个“种子模型”进行变异——初始时种子就是给定的原始模型，此后每轮迭代中，也会将前一轮**成功放大不一致**的变异结果作为新种子。然后从变异规则集合中选择一条规则应用到该种子模型，得到变异模型。接下来，通过*Keras*高级API将该模型分别在多个深度学习库后端运行，对同一输入计算输出。如果不同库产生的结果差异（例如数值不一致或异常行为）超过一定阈值，则记录为一次**不一致**。LEMON使用一个度量公式（论文中的 *D_MAD*）来量化模型在不同库上的输出差异程度，即不一致“幅度”。启发式策略会倾向于选择使得不一致幅度增大的变异操作，从而逐步**放大差异**、更容易检测到由bug引起的显著不一致。这一反馈过程持续迭代，直到达到测试预算。

LEMON的**测试判定Oracle**采用经典的**差分测试**思路：只要同一模型在不同DL库上的行为不一致，即视为揭露了一个潜在问题。这有效避免了需要预知“期望输出”的困难，将问题转化为实现之间的对比。通过20个版本的4大流行库的实验，LEMON成功发现了24个此前未知的新bug。其中一些不一致表现在模型精度差异上，有些则是在特定输入下某些库抛出异常而另一些不抛出。开发者确认了其中7个bug（1个已修复），证明这些差异确由框架实现错误导致。例如，LEMON检测到MXNet框架的`transpose`算子在处理特定维度顺序时抛出异常，而其他库正常；开发者后来修正了MXNet的实现。

作为这一领域开山之作，LEMON的贡献在于**首次提出针对深度学习库的模型级模糊测试**框架，将测试视角从模型输入层面的鲁棒性扩展到**底层算子实现正确性**。通过模型变异+跨库对比，LEMON为后续工作奠定了基础。然而，它也存在一些局限：测试覆盖的API算子种类受限于初始种子模型和变异规则（LEMON种子模型仅12个，覆盖TensorFlow约59个算子API）；另外，高阶变异虽引入更多变化但也可能降低模型有效性，需要较强的约束以确保生成模型可在各库成功运行。后续的研究（如FreeFuzz、NeuRI等）正是试图解决这些问题，提升测试覆盖和效率。

## GraphFuzz：基于计算图的深度学习推理引擎模糊测试

**GraphFuzz**（ICSE 2021）聚焦于*深度学习推理引擎*的测试。推理引擎通常指诸如TensorRT、TensorFlow-Lite、MNN等框架，用于将训练好的模型在不同硬件上高效执行。这些引擎本质上会对模型的计算图进行转换和优化，如算子融合、计算调度等。GraphFuzz的出发点是：**推理引擎本身作为软件也可能存在缺陷，需要系统化测试，以捕捉模型转换或执行中的异常**。

GraphFuzz借鉴模糊测试理念，提出一种**以计算图为核心**的数据生成和覆盖引导方法。它的思路可以概括为：“**随机构造多样的神经网络计算图作为测试输入，并定义一种算子级覆盖标准来衡量引擎对这些图的处理逻辑覆盖程度，以指导进一步生成测试**。”与LEMON不同，GraphFuzz不是依赖现有模型做变异，而是**直接生成计算图**。主要技术点包括：

- **模型生成（基于图变异）**：GraphFuzz首先定义了一个“基本块”概念，即带若干输入输出的子图片段（可以是单个算子或一组层）。然后通过**图操作变异**生成完整模型，包括：添加/删除图中的节点或连接（如加入新的算子节点、移除已有节点，或增加/切断节点之间的连边）和调整算子参数（如张量shape变异、权重参数变异）。这些变异类似于LEMON的层操作，但在更加通用的图结构层面进行。因此，GraphFuzz能构造出结构多样且复杂的模型图，触发推理引擎不同部分的逻辑。例如，它会尝试各种*拓扑结构*（分支、跳连接等）以及不同*张量维度格式*（如NCHW与NHWC）来挑战引擎的模型转换功能。为了避免生成无效模型，GraphFuzz在变异时确保基本的图合法性（如算子连接维度匹配），并提供一定的随机初始权重和随机输入数据。
- **算子级覆盖率**：GraphFuzz引入了一个专门的**Operator-Level Coverage（算子级覆盖）\**指标。直观来说，它从图论角度度量“当前测试的模型集合已经覆盖了多少种算子及其组合情况”。具体考虑了：算子类型本身、算子的入度/出度情形（即该算子在图中连接模式）、不同张量shape和参数取值等维度。可以理解为，每当生成一个模型，GraphFuzz会分析经过推理引擎处理时涉及了哪些算子逻辑（例如某特定算子实现的某分支被执行），并将其映射为覆盖分量。这个覆盖率越高，说明测试模型集合触及了引擎更多不同的功能路径。GraphFuzz将\**提升算子覆盖**作为测试目标之一。
- **MCTS搜索引导**：为了高效探索模型空间，GraphFuzz采用了**蒙特卡洛树搜索 (MCTS)** 来指导模型生成。具体而言，它将选择算子并拼接基本块的过程建模为一棵搜索树：每个节点代表一种算子选择，路径代表一个模型结构的构建方案。MCTS会模拟随机选择若干算子组合成模型，观察其带来的覆盖增益（以及是否触发异常）作为反馈，不断迭代更新节点的“价值”。这样，它逐步偏向那些能提高覆盖或发现bug的组合。相比纯随机生成，MCTS能**自适应地**探索更有前景的模型结构区域。实验显示，MCTS相比无引导随机，在算子覆盖上平均提升约6.7%，发现异常数多出近9.7个。

GraphFuzz的测试执行采用两类*oracle*判定： 其一，当引擎在处理生成模型时如果发生**模型转换失败**（如导入模型时抛出错误）或**推理运行失败**（如崩溃、NaN/Inf输出），则直接视为找到bug。其二，对于转换成功且正常运行的模型，GraphFuzz还会进行**输出比对**：将模型在不同环境下运行结果进行比较。例如，他们在MNN推理引擎上用CPU和ARM两种硬件运行同一模型，或对比引擎优化前后结果，若输出显著不一致就标记为异常。这一点类似LEMON的差分，但是在同一引擎的不同配置下进行。通过以上手段，GraphFuzz在MNN引擎上发现了超过40个不同异常，归类为三种类型：(1) 模型转换失败；(2) 推理过程中崩溃或错误；(3) 输出数值差异（如精度损失过大）。其中一些属于严重问题，例如某些算子在特定数据格式下返回错误结果等。

GraphFuzz的创新在于**将神经网络测试提升到了“生成计算图+覆盖引导”这一抽象层**。它证明了在没有现成模型的情况下，也可以通过智能搜索生成高质量测试模型来评估推理引擎的健壮性。实验证明，GraphFuzz生成的模型相比以前依赖手工模型或简单变异的方法，能触发更多隐藏的问题。其局限在于：实现MCTS和覆盖度量需要对引擎有深入了解和插桩支持，针对不同引擎可能需要调整。此外，GraphFuzz主要针对推理阶段，对训练过程相关的API未涉及。这方面FreeFuzz等工作通过API级fuzz进行了补充。总体而言，GraphFuzz为**结构化探索深度学习系统空间**提供了有力范式，后来NeuRI也在其基础上进一步结合了符号约束推理来提高模型生成的有效性和多样性。

## FreeFuzz：利用开源数据的深度学习库API模糊测试

以往的模型级测试（如LEMON）存在**覆盖面有限**的问题：它们依赖少量预置模型和手工变异规则，难以触及深度学习库成百上千的API。**FreeFuzz**（ICSE 2022）针对这一痛点，提出了全新的思路：**将测试粒度下沉到单个API调用级别，通过大规模挖掘“野生”代码来获取API调用样本，并进行参数变异以发现异常**。这种方法本质上类似于传统软件的单元测试，将复杂模型拆解为库函数的逐一测试，从而显著扩大测试覆盖。

FreeFuzz的整体框架包括**收集、履历提取、变异、差分检测**四个阶段。下面分别说明：

- **开源用例收集**：FreeFuzz从三类来源自动收集深度学习库的使用案例：（1）官方*文档示例*代码，例如PyTorch或TensorFlow文档中的API用法片段；（2）库自身的*单元测试*代码，即开发者为保证库正确性编写的测试（这往往涵盖了不少典型用法）；（3）*开源DL模型*代码，从GitHub等获取真实项目中构建模型的脚本。通过这些来源，FreeFuzz获得了一个覆盖上千API的代码语料。接下来，FreeFuzz运行所有这些代码片段，**动态截获每个API的调用信息**。具体而言，它使用插桩记录下每次API调用时**各参数的类型、值以及tensor维度**等信息。比如，遇到一行调用`torch.nn.Conv2d(16, 33, (3,5), stride=(2,1))`，FreeFuzz会记录下参数`in_channels=16`（类型int）、`out_channels=33`、`kernel_size=(3,5)`（类型tuple）、`stride=(2,1)`（类型tuple）等，以及返回的层对象要求输入tensor形状等。通过运行大量真实代码，FreeFuzz构建了一个**参数值数据库**：针对每个API，收集到一组可能的参数取值及类型分布。与静态文档相比，这些动态信息非常宝贵——它告诉我们“在真实场景下某API常见怎样的调用方式”。统计结果表明，FreeFuzz成功为1158个API收集到了有效参数组合，而之前LEMON一类方法在TensorFlow中仅涉及不到60个API。这奠定了宽覆盖的基础。
- **参数模糊变异**：有了每个API的“典型参数空间”之后，FreeFuzz针对每个API进行独立的模糊测试。测试时，它会从数据库提取该API的一组有效参数组合作为起点（这些组合保证API能跑通，不会触发无效调用错误）。然后应用多种**参数变异策略**来产生新测试用例：一是*类型变异*（Type Mutation），即改变参数的数据类型，例如某参数本来是int就尝试换成float或张量等，看是否处理得当；二是*随机值变异*，对数值型参数随机扰动或赋予极端值；三是*数据库值重组*（Database Value Mutation），即利用之前收集的值组合，在不同但相似的API间共享参数值，例如Conv2d和Conv3d可以交换部分参数。通过这些变异，FreeFuzz为每个API生成大量不同调用方式（相当于构造各种“怪异”的参数输入）。每次变异后，它调用该API执行，并采用如下**判定机制**：
  - 如果API在某组参数下**抛出未预期的异常**（如段错误、断言失败），则视为找到bug。需要注意，有些异常可能是参数无效导致，这里FreeFuzz依赖之前收集的有效值尽量避免无意义错误，并进一步结合形同测试降低误报。
  - 如果API返回结果，但结果在不同执行环境下不一致，亦视为bug。FreeFuzz主要采用两种差分：其一是**多后端差分**，如对同一TensorFlow API在CPU和GPU后端运行结果比对，或PyTorch的不同设备/模式比对。其二是**元变换测试**，例如对一些数学函数，加上等效变换（如`sin(x)^2+cos(x)^2`应恒等于1）前后结果应一致，或者对神经网络层前后shuffle输入等，输出应满足预期关系。如果出现违反预期的差异，即报告潜在问题。

FreeFuzz在PyTorch和TensorFlow上进行了大量实验，结果令人瞩目：它成功探索了1158个API（9倍于先前方法），在几个月内提交了49个新bug报告，其中38个被开发者确认是之前未知的问题，已有21个被修复。这些bug涉及范围广泛，例如：某些API在特定类型输入时返回错误结果，某些算子在GPU模式下精度问题，甚至内存泄漏等。FreeFuzz证明了**API级模糊测试的巨大威力**。相比模型级测试，API级测试**更细粒度、更高效**。一方面，每个API调用独立执行，避免了整网路执行的高开销（无需反复跑长时间训练或推理流程）；另一方面，聚焦单API也减少了浮点误差积累导致的误报，可更容易定位问题根源。正如作者所说，模型级测试类似于系统测试，而API级更像单元测试，两者相辅相成。

需要指出的是，FreeFuzz通过大规模“免费午餐”式利用开源资源，实现了测试输入的**零成本获取**。这在深度学习测试领域打开了一条新路：不再完全依赖研究者人工设计变异策略，而是**让海量现有代码来提示可能的有效输入模式**。当然，FreeFuzz也有局限，比如收集到的用例仍以常规用法为主，可能覆盖不到非常规组合；而NeuRI进一步探讨了如何自动推理算子间约束生成更复杂用例。总的来说，FreeFuzz极大拓展了测试覆盖，被认为提供了一个强有力基线。后续有研究在此基础上结合历史执行信息或改进输入生成策略，以发现更多棘手bug。

*FreeFuzz的流程概览：首先从文档、测试和开源模型中收集代码，运行并记录API调用及参数（阶段①②）；然后基于收集的参数空间进行多策略变异生成API调用（阶段③），最后通过差分测试和形同验证检测错误（阶段④）。该方法显著扩大了可测试API数量，从而覆盖更多深度学习库代码。*

## EAGLE：生成等价计算图来测试深度学习库

深度学习库的某些功能往往只有单一实现，缺乏可供直接对比的另一库实现（这是差分测试的前提）。**EAGLE**（ICSE 2022）巧妙地提出了**“不同路径实现同一功能”的等价测试**思路，以此绕过需要多库的限制。其核心是：**设计一系列等价变换规则，将同一计算功能用两种不同的计算图表示，然后在同一库中分别执行，比较输出是否一致**。如果结果不一致，就说明库内部在这些等价情况下处理不统一，极可能存在bug。

EAGLE首先**人工定义**并程序化实现了16条通用的“等价图”规则。这些规则覆盖常见的深度学习图等价类别，如：**计算优化等价**（开启/关闭某种图级优化应不影响结果）、**API冗余等价**（某些复合API等价于多个基础API组合）、**数据结构等价**（不同数据类型或容器存储相同内容）、**数据格式等价**（如张量的不同存储布局转换）、**逆操作等价**（某操作的逆操作组合成恒等变换）、**模型评估等价**（训练模式与评估模式结果在特定条件下应一致）等[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=In this paper%2C we make,detects bugs in DL libraries)[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=to test DL libraries,Using the 16)。每条规则抽象描述一类情形下，两种计算流程应该给出相同结果。

为更具体地说明，论文以**RNN时间序列格式**的等价性为例：很多RNN类函数都有一个参数控制输入tensor的格式是“[批次, 时间步]（batch-major）”还是“[时间步, 批次]（time-major）”。两种格式只是张量转置关系，本质计算应等价。EAGLE据此定义规则：“对于任意RNN层，当`time_major=False`和`True`时，通过在输入和输出处各加一个转置操作，这两种配置的网络应输出相同结果”。如图所示，一个双向RNN层在batch-major模式下需要将反向RNN输出按时间维逆序（正确实现应该对时间维reverse）；但在time-major模式下，有一版本TensorFlow错误地对**批次维**做了reverse，导致输出与前一种实现不一致。通过构造这两个等价的计算图并对比输出，EAGLE成功捕捉到了这个TensorFlow中的缺陷。类似地，其他规则涵盖如：“一个DepthwiseConv2D等价于多个Conv2D并联”“带或不带某优化旗标（如`tf.function`）执行应等效”等等[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=Figure 2 presents the overview,section describes the equivalence rules)。

接下来，EAGLE的测试过程包含三步[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=output 𝑂0,to)：

1. **等价规则应用**：针对每条抽象等价规则，EAGLE会参考TensorFlow和PyTorch官方文档，找出适用该规则的一组API函数[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=real,pairs of TensorFlow equivalent graphs)[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=In total%2C we design a,input%2C configuration)。然后为每个API构造一对等价的具体计算图（Graph1和Graph2）。例如RNN规则会生成一个使用`Bidirectional` RNN层的Graph1和一个使用基础RNN但不同参数配置的Graph2。为了增加测试强度，每对等价图会随机选择多组不同的输入数据和配置参数来运行（论文中每对图运行了400组输入+配置）[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=Equivalent graph construction%3A Once we,generate valid input based on)。这些输入会根据API约束随机生成，确保在有效范围内[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=each is tested with 400,from a pair of concrete)。
2. **等价执行对比**：将每组输入分别喂给等价的Graph1和Graph2，在**同一个深度学习库**上执行，得到输出O1和O2[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=equivalent graphs ,section describes the equivalence rules)。因为两图表达的是数学上等价的功能，正确情况下输出应当*完全一致*（或在浮点误差容许范围内相等）[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=The first step is to,result in slightly different outputs)。如果EAGLE检测到O1与O2存在超出允许误差的差异，则判定发现了一处**不一致bug**。这样的bug通常意味着库的某个实现分支有错误。例如前述RNN案例，就是因TensorFlow对time-major参数分支实现有误造成的不同结果。
3. **结果分析报告**：EAGLE将检测到的不一致加以分类汇总，并提取对应的底层问题（比如涉及哪个API，在什么配置下出错）形成bug报告提交给开发者。在评估中，EAGLE针对TensorFlow和PyTorch各测试了数千对等价图，最终检测出25个bug（TensorFlow 18个，PyTorch 7个），其中13个此前未知。这些bug涵盖了所有6类等价规则，说明各类等价性都有可能出现实现不一致问题。例如，所有TensorFlow的双向RNN层在time-major模式下均存在类似bug；再如PyTorch在某优化关闭情况下结果与开启时不同，等等。开发者确认了许多报告的有效性。

EAGLE的突出贡献在于**将差分测试拓展到单库内部的等价执行**，提出了“一库双图”的新颖测试维度。相比以往需要两个框架比较的方法（如CRADLE、LEMON），EAGLE不受限于他库实现是否存在，适用范围更广。此外，16条等价规则的设计凝聚了对深度学习API实现的深入理解，具有通用性，覆盖了上千个API[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=real,pairs of TensorFlow equivalent graphs)。这种**基于领域知识的等价变换**也属于*元模糊测试*的一种形式，在测试其它系统时亦有借鉴意义。需要注意的是，EAGLE目前规则主要人工制定，可能尚未穷尽所有等价情况；而NeuRI等则探索自动归纳规则。尽管如此，EAGLE已经成功找出了许多微妙的非崩溃错误，尤其是在训练与推理、不同优化之间的一致性问题上。对于保障深度学习模型的可信执行，这类测试非常关键。例如，一个模型如果在不同运行模式下输出不一致，可能导致部署隐患。EAGLE正是捕捉此类问题的有力工具。

*EAGLE方法概览：通过查阅文档定义通用等价规则（步骤1），针对每条规则在单个库中生成一对等价的计算图并施加各种输入（步骤2），然后运行并交叉对比输出检测不一致（步骤3）[jiannanwang.github.io](https://jiannanwang.github.io/files/eagle-icse22.pdf#:~:text=Figure 2 presents the overview,section describes the equivalence rules)。这种无需多库的差分测试能揭示单库实现中的隐藏bug。*

## NeuRI：通过归纳规则推理实现DNN生成多样化

随着深度学习库和**编译器**的演进（如PyTorch 2的JIT编译器、TensorFlow XLA等），测试不仅要涵盖普通算子API，还需关注这些新组件。**NeuRI**（ESEC/FSE 2023）致力于**自动生成多样且有效的深度学习模型**来测试现代DL系统，包括编译器在内。它名字中的“NeuRI”取义“Neural Rule Inference”，反映其特色：通过*归纳推理*（inductive synthesis）自动学习算子使用规则，以实现**全面的模型生成**。

以往工具（如LEMON、GraphFuzz）在生成模型时面临两难：要么为了简单而限制支持的算子种类和连接方式，导致模型**多样性不足**，覆盖不了众多API；要么尝试任意拼接算子又经常产生**无效模型**（因张量shape或参数不符而报错）。NeuRI正是为破解这一难题而生。它提出了一个三步流程：

1. **收集API调用轨迹**：NeuRI首先从各种来源收集深度学习库/API的调用序列（trace），包括**有效**的和**无效**的调用例子。来源如：官方示例、单元测试、开源模型，以及通过随机尝试收集到的一些失败调用等。这些trace可以视为“算子如何串联使用”的实例库。
2. **归纳规则推理**：接下来，NeuRI运用*归纳程序合成*技术对收集到的调用轨迹进行分析，总结出算子之间的**约束规则**。简单说，它试图推理出“什么条件下某组算子拼接是有效的”。例如，某Pooling层要求前一层输出的尺寸大于其池化窗口，否则连接会失败；又如，某些算子参数（stride、kernel等）与输入张量shape之间存在数学关系约束。NeuRI将这些隐含的有效性规则用形式化的表达式表示出来[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=shape propagation rules can be,an operator rule via inductive)[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=program synthesis%2C i,To infer)。它的做法是：把规则的推理建模为在一个定义好的表达式空间中搜索，使表达式能解释所有已知的成功/失败案例差异[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=We now explain how to,Symbols from 𝐼 and 𝐴)[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=program synthesis%2C i,in the “partial operator” paragraph)。具体采用了一套算术表达式文法（加减乘除、min/max、mod等）[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=whose bodies are oftentimes arithmetic,an operator rule via inductive)，对每个算子的输入/属性，合成出约束谓词和shape推导公式，使之在已知调用记录上成立[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=computing corresponding output dimensions,are predicates of equalities and)[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=∃expr∈ E%2C ∀⟨𝐴 ★%2C 𝐼★%2C𝑂★⟩,the loop for the next)。这实际上是枚举+验证的过程，NeuRI通过大量优化（如剪枝、不变量假设等）来令推理在可行时间内完成[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=program synthesis%2C i,in the “partial operator” paragraph)[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=constructs small terms (e,Formula 3)。最终，NeuRI得到了一组**算子有效性规则**（包括输入约束和输出shape推导规则）[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=We now explain how to,Symbols from 𝐼 and 𝐴)[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=computing corresponding output dimensions,are predicates of equalities and)。这些规则可以覆盖上百种算子，远超人为手工定义的少数规则。值得注意的是，此步骤有点类似EAGLE的规则设计，但NeuRI是**自动归纳**出来的，减少了人工参与。
3. **混合模型生成**：有了规则库，NeuRI进入测试用例生成阶段。它采取**符号+具体（混合）生成**策略。所谓符号，即NeuRI可以放入“占位符算子”根据规则随意拼接，而不立即指定具体是哪种算子，只要满足类型和shape约束即可；具体指实际可执行的特定算子。NeuRI先根据想要覆盖的算子类型，利用推理规则符号地拼出模型骨架，同时确保每一步连接、每个参数都满足前述约束（这样保证模型有效）[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=Reports Figure 3%3A Overview of,Specifically%2C we can)[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=shape propagation rules can be,an operator rule via inductive)。在此过程中，它会随机挑选算子并根据规则检查兼容性，直到拼出一定深度和复杂度的模型图。此外，为了兼顾真实感，NeuRI还会掺入从真实模型中截取的子图片段（concrete ops）。最终得到的模型包含了丰富多样的算子及组合关系，而且**保证通过规则校验**为有效模型，不会一生成就报参数错误。NeuRI将这些模型转为实际可运行的GraphIR，送入目标库或编译器进行测试[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=Concolic Op Insertion GraphIR Models,Inconsistent Results Runtime Error Sanitizer)。

NeuRI的测试oracle与GraphFuzz类似，也是多管齐下：它会监测模型编译或运行时是否出现异常崩溃，或者对比编译器与解释器模式结果是否不一致（例如PyTorch 2编译与直跑结果差异）[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=Concolic Op Insertion GraphIR Models,Inconsistent Results Runtime Error Sanitizer)[kristoff-starling.github.io](https://kristoff-starling.github.io/pubs/fse2023_neuri.pdf#:~:text=Inconsistent Results Runtime Error Sanitizer,4 Oracles Bug Reports)。通过极大丰富的模型输入，NeuRI取得了惊人的效果：在4个月内对TensorFlow和PyTorch发现了100个新bug，其中81个已被确认或修复！更重要的是，NeuRI显著提高了测试覆盖——相比当时最先进的模型级fuzzer（FreeFuzz/LEMON等），NeuRI在TensorFlow和PyTorch上分别多覆盖了24%和15%的分支。很多开发者反馈NeuRI找到的错误用例“质量很高”“在实际场景中很常见”。

可以说，NeuRI将深度学习库测试推进到了一个新高度。它**自动学会了算子如何正确组合**，从而能生成前所未有多样复杂的模型来“难为”库，实现了测试输入多样性和有效性的双赢。这是对之前基于人工经验设计变异（LEMON）或仅从已有用例挖掘（FreeFuzz）的重大升级。当然，NeuRI的实现也相当复杂，它借鉴了程序合成领域的技术，不同算子的规则推理需要大量计算。不过其成果证明了自动推理在测试上的巨大潜力。在未来，NeuRI的思想还可推广到其它框架和更多算子，不再局限于手工规则或有限样本，真正实现**全面的DL系统自动测试**。总之，NeuRI为深度学习软件测试提供了*智能化*的新范式，也是目前该领域最先进的研究之一。