# DI-star 智能体训练机制深度分析报告

## 1. 项目概览

**DI-star** 是专为《星际争霸II》（StarCraft II）打造的大规模分布式强化学习训练平台。该项目旨在训练出具备宗师级水平的游戏AI，其技术架构深度借鉴了 DeepMind 的 AlphaStar 设计，并针对开源社区和有限计算资源进行了适配与优化。

项目核心特点：
- **完整训练管线**：支持监督学习（SL）预训练和强化学习（RL）微调。
- **分布式架构**：采用 Coordinator-League-Actor-Learner 的经典分布式RL架构，支持大规模并行训练。
- **League 自我博弈**：通过构建由多种角色（Main, Exploiter等）组成的联盟，实现持续进化。
- **多模态观察空间**：完美处理SC2复杂的观察数据，包括地图（Spatial）、单位（Entity）和统计信息（Scalar）。
- **自回归动作空间**：采用分层的动作分解策略，精确控制游戏中的每一步操作。

---

## 2. 状态空间（Observation Space）设计

DI-star 的状态空间设计直接继承自 AlphaStar 的观察编码器，将复杂的SC2原始观测数据（Raw Data）系统性地编码为三大类张量：**空间特征（Spatial）**、**标量特征（Scalar）** 和 **实体特征（Entity）**。这种设计旨在保留环境的所有关键信息，同时通过归一化和嵌入（Embedding）处理，使其适用于深度神经网络。

### 2.1 空间特征（Spatial Information）

空间特征主要来源于SC2的 小地图（Minimap） 和 特效层（Feature Layers）。在 `distar/agent/default/lib/features.py` 中定义，并在 `SpatialEncoder` 中进行处理。

**核心组成及形状**：
- **基础地图层** (`SPATIAL_SIZE = [152, 160]`, y, x)：
  - `height_map`: 地形高度图。
  - `visibility_map`: 视野可见性（隐藏/迷雾/可见）。
  - `creep`: 虫族菌毯分布。
  - `player_relative`: 单位所属关系（自己/盟友/中立/敌人）。
  - `alerts`: 警报信息（如受到攻击）。
  - `pathable`: 是否可通行。
  - `buildable`: 是否可建造。
- **特效层** (`effect_*`): 记录如灵能风暴（PsiStorm）、核打击标记（NukeDot）、 liberator 区域等技能特效的位置，以稀疏坐标列表形式存储（`EFFECT_LEN=100`）。

**预处理与编码**：
- 空间特征首先通过 `Feature.unpack` 从原始协议缓冲区解析为 NumPy 数组。
- 在 `SpatialEncoder` 中，基础地图层经过 One-hot 编码（如 `player_relative` 有5种状态，被编码为5个通道）和归一化（如 `height_map / 256`）。
- 实体位置信息通过 `scatter_connection` 被处理为稀疏的 `scatter_map`，并与基础地图层拼接。
- 整体通过卷积层进行下采样（Downsample），提取空间上下文特征。

### 2.2 标量特征（Scalar Information）

标量特征是对全局游戏状态的高度概括，避免了网络需要从原始像素中推断宏观数据。定义于 `SCALAR_INFO` 中，由 `ScalarEncoder` 编码。

**核心组成**：
- **玩家统计 (`agent_statistics`)**: 包含水晶矿、瓦斯、人口、军队数量等10维数据。经过 `log(x+1)` 变换以减小数值范围。
- **单位统计**:
  - `unit_counts_bow`: 己方单位的词袋模型（Bag-of-Words），记录各类单位的数量。
  - `unit_type_bool`: 己方单位类型的布尔掩码（表示是否拥有某种单位）。
  - `enemy_unit_type_bool`: 侦察到的敌方单位类型布尔掩码。
  - `unit_order_type`: 己方单位正在执行的命令类型分布。
- **经济与科技**:
  - `upgrades`: 已研究的升级项目（90维 One-hot）。
  - `home_race` / `away_race`: 双方种族信息（One-hot）。
- **时序信息**:
  - `time`: 当前游戏帧（Game Loop），使用位置编码（Transformer 风格）进行嵌入。
  - `last_action_type`, `last_delay`, `last_queued`: 上一步的执行动作类型、延迟和是否队列，作为动作的条件依赖。
- **目标向量 (Z / Teacher Features)**:
  - `beginning_order` & `bo_location`: 从人类回放提取的目标建造顺序（BO）及其位置，共20步（`BEGINNING_ORDER_LENGTH=20`）。
  - `cumulative_stat`: 目标累积统计（如应该建造哪些关键建筑/单位），167维布尔向量。
- **其他**: 如 `enemy_agent_statistics`（敌方统计，用于价值网络）等。

**预处理与编码**：
- 大部分标量特征根据其语义进行 One-hot 编码或全连接层（FC）嵌入。
- `beginning_order` 使用特殊的 `BeginningBuildOrderEncoder`，包含动作嵌入、序列位置嵌入和二进制位置编码，通过 Transformer 提取序列特征。

### 2.3 实体特征（Entity Information）

实体特征是对游戏中每一个可见单位的详细描述，是SC2中最丰富的信息来源。每个单位被视为一个独立的“实体”（Entity），最多处理512个（`MAX_ENTITY_NUM=512`）。由 `EntityEncoder` 编码。

**核心组成 (ENTITY_INFO)**：
包含超过30个字段，精确描述单位的静态与动态属性：
- **基础属性**: `unit_type`（类型）、`alliance`（所属关系）、`x`, `y`（坐标）、`display_type`（可见/快照/隐藏）。
- **状态**: `health_ratio`, `shield_ratio`, `energy_ratio`（生命值等比例，已完成归一化）。
- **动态**: `build_progress`（建造进度）、`weapon_cooldown`（武器冷却）、`order_length`（指令队列长度）。
- **指令队列**: `order_id_0` .. `order_id_3`（当前正在执行的最多4个指令的ID）、`order_progress_0`, `order_progress_1`（指令进度）。
- **增益/附加**: `buff_id_0`, `buff_id_1`（身上带的Buff，如中毒）、`addon_unit_type`（建筑附加组件）。
- **上下文**: `last_selected_units`, `last_targeted_unit`（标记该单位是否在上一步被选中或作为目标）。
- **资源**: `mineral_contents`, `vespene_contents`（对于矿脉/气泉，存储资源量）。

**预处理与编码**：
- `unit_type`, `alliance`, `buff_id` 等离散ID使用固定的 One-hot 嵌入矩阵。
- `x`, `y` 坐标使用二进制编码（`get_binary_embed_mat`），共11位，覆盖 0-2047 的范围。
- 所有实体信息拼接后，形成一个 `(Batch, MAX_ENTITY_NUM, input_dim=997)` 的张量。
- 通过 `EntityEncoder` 中的 **Transformer**（多头注意力网络）进行处理，使单位之间能够相互感知（例如，一个单位可以“看到”附近的敌人）。
- 最后通过 `Attention Pooling` 或 `Mean Pooling` 将变长的实体列表压缩为固定长度的向量（`embedded_entity`）。

### 2.4 观察空间总结表

| 类别 | 特征名 | 形状 (Shape) | 描述 | 编码方式 |
| :--- | :--- | :--- | :--- | :--- |
| **Spatial** | `height_map` | `[152, 160]` | 地形高度 | 归一化 (除以256) |
| | `player_relative` | `[152, 160]` | 单位阵营 | One-hot -> `(B, 5, H, W)` |
| | `creep` | `[152, 160]` | 菌毯 | One-hot -> `(B, 2, H, W)` |
| | `effect_*` | `[100]` (稀疏列表) | 技能特效坐标 | Direct Index |
| **Scalar** | `agent_statistics` | `[10]` | 资源/人口等 | Log Transform + FC Embed |
| | `upgrades` | `[90]` | 已研究升级 | One-hot + FC Embed |
| | `beginning_order` | `[20]` | 目标建造顺序 | Transformer Embed |
| | `time` | `[1]` | 游戏时间步 | Positional Encoding |
| | `last_action_type` | `[1]` | 上一步动作 | One-hot Embed |
| | `enemy_unit_type_bool`| `[260]` | 敌方单位类型 | One-hot Embed |
| **Entity** | `unit_type` | `[512]` | 单位类型ID | One-hot Embed |
| | `alliance` | `[512]` | 单位阵营ID | One-hot Embed |
| | `health_ratio` | `[512]` | 生命值比例 | Float (0-1) |
| | `x`, `y` | `[512]` | 地图坐标 | Binary Embed (11-bit) |
| | `order_id_0` | `[512]` | 当前动作ID | One-hot Embed |
| | `buff_id_0` | `[512]` | Buff ID | One-hot Embed |

---

## 3. 动作空间（Action Space）设计

DI-star 的动作空间高度结构化，采用自回归（Auto-regressive）和分层（Hierarchical）的设计，将复杂的SC2操作分解为一系列条件依赖的子动作。这种设计极大地降低了动作组合的空间爆炸问题。

整个策略网络输出被分解为 **6个独立的动作头（Action Heads）**，定义在 `distar/agent/default/model/policy.py` 中。

### 3.1 动作类型（Action Type）

- **维度**：327 (`NUM_ACTIONS`)。
- **描述**：从预定义的宏观动作（Macro-actions）列表中选择一个。这些动作涵盖了移动、攻击、建造建筑、训练单位、研究科技、施放技能等几乎所有核心操作。
- **设计**：这是一个典型的 AlphaStar 设计，使用约300-500个高级动作而非原始的SC2 API函数（数千个）。每个 `action_type` 内部定义了所需参数（如是否需要目标、是否需要选中单位）。
- **输出**：`ActionTypeHead` 基于 LSTM 输出和 Scalar 上下文生成一个 327 维的逻辑斯谛向量（Logit）。

### 3.2 延迟（Delay）

- **维度**：128 (`MAX_DELAY` = 127)。
- **描述**：决定在执行当前动作前，AI应该等待多少帧。
- **作用**：模拟人类的反应时间和 APM（每分钟操作数）限制，防止AI以非人类的频率疯狂操作。
- **输出**：`DelayHead` 生成一个 128 维的 Logit。在推理时用于控制 APM。

### 3.3 队列（Queued）

- **维度**：2。
- **描述**：决定当前动作是直接执行（`queued=0`， Shift 未按下），还是加入单位指令队列（`queued=1`， Shift 按下）。
- **输出**：`QueuedHead` 生成一个 2 维的 Logit。

### 3.4 选中单位（Selected Units）

- **维度**：动态，最多选择 64 个单位（`MAX_SELECTED_UNITS_NUM`）。每个选择从 `MAX_ENTITY_NUM + 1` (512+1，包含一个“结束”标记) 中决定。
- **描述**：使用一个内部 LSTM，自回归地逐个选择单位，直到选择“结束”标记。
- **设计**：
  - 这是整个动作空间中最复杂的部分，使用了 `SelectedUnitsHead`。
  - 它通过计算 Query（来自 LSTM 和先前选择的单位信息）和 Key（来自实体编码）的点积，生成注意力分数，从而决定下一个选择哪个单位。
  - 已选中的单位会被掩码（Mask）掉，防止重复选择。
- **输出**：一个形状为 `(selected_units_num, MAX_ENTITY_NUM + 1)` 的 Logit 序列。

### 3.5 目标单位（Target Unit）

- **维度**：`MAX_ENTITY_NUM` (512)。
- **描述**：当动作类型需要一个目标单位时（如攻击单位、治疗单位），从当前可见的512个实体中选择一个。
- **输出**：`TargetUnitHead` 生成一个 512 维的 Logit。

### 3.6 目标位置（Target Location）

- **维度**：`SPATIAL_SIZE[0] * SPATIAL_SIZE[1]` = `152 * 160` = 24,320。
- **描述**：当动作类型需要一个地图坐标时（如移动、建造建筑、技能施放），在完整地图上选择一个像素点。
- **设计**：`LocationHead` 使用 U-Net 风格的结构，结合 LSTM 输出和 `SpatialEncoder` 的跳连（Skip Connections）特征图，通过上采样生成一个与地图分辨率一致的 Logit 热图，最终展平为 24320 维。
- **输出**：形状为 `(24320,)` 的 Logit 向量。

### 3.7 动作掩码（Action Masking）

在 `agent.py` 的 `_post_process` 和相关逻辑中，会根据 `action_type` 决定哪些动作参数头是有效的（Valid），哪些是无效的。例如，如果 `action_type` 是 `no_op`（什么都不做），那么 `delay`, `queued`, `selected_units` 等头的输出实际上会被忽略。这种掩码机制进一步约束了动作空间，使其符合SC2的游戏规则。

---

## 4. 奖励函数（Reward Function）设计

DI-star 的奖励函数采用了多目标（Multi-objective）设计，结合了稀疏的胜负信号和稠密的模仿学习/过程奖励，这同样是对 AlphaStar 奖励设计的忠实复现。

### 4.1 训练模式与损失项

在 `distar/agent/default/rl_training/rl_loss.py` 中，总损失函数是一个复合体，包含以下几个核心部分：

1.  **策略梯度损失 (Policy Gradient Loss)**：使用 **V-Trace** 算法进行离线策略修正（Off-policy correction）。通过计算重要性采样比率 `rho = target_policy / behavior_policy`，并利用 `clipped_rhos` 来获得 `vtrace_advantages`。这允许Learner在使用旧策略采样（Actor端）的数据进行高效更新。
2.  **UPGO 损失 (Upgoing Policy Gradients)**：一种特殊的策略梯度变体，利用 `upgo_returns` 来提供更稳定的策略梯度估计，帮助策略更快地倾向高价值动作。
3.  **Critic 损失 (Critic Loss / TD-Lambda Loss)**：使用时序差分（TD）结合 Lambda 回报（`generalized_lambda_returns`），计算每个价值头的均方误差（MSE）。
4.  **熵正则化 (Entropy Loss)**：鼓励策略保持探索性，防止过早收敛到局部最优。
5.  **KL 散度约束 (KL Loss)**：限制当前策略与监督学习预训练的教师模型（Teacher Model）或先前策略（Successive Model）之间的距离，确保强化学习过程不会“忘记”基础的游戏知识，保持策略稳定。

### 4.2 多目标奖励头（Reward Heads）

不同于传统的单一胜利奖励，DI-star 为 **每个价值网络（Value Baseline）** 定义了独立的奖励来源，实现在 `distar/agent/default/agent.py` 的 `_update_fake_reward` 和 `collect_data` 中。

| 奖励头 | 英文名 | 描述 | 稀疏/稠密 | 计算公式/来源 |
| :--- | :--- | :--- | :--- | :--- |
| **胜负** | `winloss` | 游戏最终胜利/失败 | **稀疏** | 游戏结束时返回 `+1` / `-1`，经缩放。 |
| **建造顺序** | `build_order` | 模仿目标 BO 的准确程度 | **稠密** | 基于行为BO与目标BO的莱文斯坦距离（Levenshtein Distance）差分奖励。公式：`new_dist - old_dist`。 |
| **累积统计** | `built_unit` | 模仿目标累积统计的准确程度 | **稠密** | 基于行为累积统计与目标累积统计的汉明距离（Hamming Distance）差分奖励。随游戏时间衰减（Time Factor）。 |
| **战斗** | `battle` | 基于战损的战斗表现 | **稠密** | `己方击杀价值 - 敌方击杀价值`。`BattleScore = Killed_Mineral + 1.5 * Killed_Vespene`。 |
| **效果/升级** | `effect` / `upgrade`| 特定科技或技能的使用 | **稠密/稀疏** | 在配置文件中存在，实际权重常为0。 |

### 4.3 关键奖励机制详解

#### 4.3.1 Z 奖励（模仿学习奖励）
这是 DI-star 奖励系统中最具特色的部分。
- **Z 向量提取**：在每一局游戏开始时（`agent.reset`），系统会根据当前地图、种族和出生点，从一个预计算的 JSON 文件（`3map.json` 等）中，随机抽取一个人类高手回放中的宏观策略模板，称为 **Z**。
- **Z 组成**：
  1.  **目标建造顺序 (`beginning_order`)**：前20个关键建筑/单位的建造指令序列。
  2.  **目标累积统计 (`cumulative_stat`)**：本局中应该完成的关键科技和单位集合的布尔向量。
- **行为记录**：在游戏过程中，`agent` 会实时记录自己的行为 BO 和 CUM。
- **奖励计算**：
  - **BO 奖励**：使用 `levenshtein_distance`（或带位置的 L2 距离）计算当前行为 BO 与目标 BO 的差异。当新建造一个属于 BO 的关键建筑/单位时，如果距离减小，就获得正奖励。
  - **CUM 奖励**：使用 `hamming_distance` 计算当前行为 CUM 与目标 CUM 的差异。每当完成一个目标 CUM 中的动作，且匹配度提高时，获得正奖励。并带有 `time_factor`（越后期奖励权重越低）。
- **作用**：这些 Z 奖励极大地密集化了奖励信号，引导 AI 在游戏中学习人类的宏观策略（开局），即使没有立即的胜负反馈，AI 也能获得有效的学习信号。

#### 4.3.2 战斗奖励（Battle Reward）
- 为了鼓励AI积极进攻和进行有效交战，系统引入了战斗奖励。
- `compute_battle_score`: 根据游戏内统计的“击杀矿物价值”和“击杀瓦斯价值”计算得分，其中瓦斯权重更高（1.5倍），因为高级单位价值更高。
- `battle_reward = (current_bs - last_bs) - (current_opp_bs - last_opp_bs)`：计算己方与敌方在战损得分上的差值。
- 这个奖励提供了比胜负信号更即时的反馈，鼓励AI在战斗中取得优势。

#### 4.3.3 胜负奖励（Win/Loss Reward）
- 这是最核心、最稀疏的奖励。在游戏结束时，根据胜负情况返回。
- 在 `rl_user_config.yaml` 中，`baseline.winloss: 10.0` 和 `pg.winloss: 1.0` 显示了其价值网络权重和策略梯度权重。

---

## 5. 训练架构与核心算法

### 5.1 分布式训练架构

DI-star 采用模块化的分布式架构，主要组件定义在 `distar/ctools/worker/` 和 `distar/actor/` 中。

- **Coordinator (协调器)**：负责 Actor 和 Learner 之间的任务调度与心跳管理。
- **League (联盟管理器)**：维护 Active Players（如 MainPlayer, Exploiter）和 Historical Players。根据胜率（Payoff）动态生成对抗任务（Self-play, PFSP等）。
- **Actor (执行器)**：运行 SC2 游戏实例，加载模型进行推理，收集 `(obs, action, reward)` 轨迹数据，并发送给 Learner。
- **Learner (学习器)**：接收来自多个 Actor 的轨迹数据，计算损失并更新模型权重，然后将新模型同步回 Actor。

### 5.2 核心训练算法

DI-star 的 RL 算法是对 AlphaStar 的 V-Trace + PPO + UPGO 组合的实现。

1.  **V-Trace (V-trace)**：
    - 用于解决 Actor-Learner 之间的策略延迟（Staleness）问题。Actor使用的模型可能落后于Learner的当前模型。
    - 通过 `clipped_rhos` 计算修正后的优势函数 `vtrace_advantages`，使得旧数据仍可用于策略更新。
2.  **PPO (Proximal Policy Optimization)**：
    - 体现在 `policy_gradient_loss` 中，通过重要性采样比率 `rho` 限制策略更新的步长，防止策略崩溃。
3.  **UPGO (Upgoing Policy Gradients)**：
    - 用于计算 `upgo_loss`，通过 `upgo_returns` 提供比 V-Trace 更乐观的价值估计，加速策略向高回报方向演化。
4.  **Actor-Critic**：
    - 模型同时输出策略（Actor）和价值（Critic）。价值网络包含多个并行的 Baseline Head，分别对应 `winloss`, `build_order`, `built_unit`, `battle` 等不同奖励。

### 5.3 League 自我博弈

在 `distar/ctools/worker/league/league.py` 中实现：
- **玩家类型**：
  - `MainPlayer (MP)`: 主智能体，目标是提升综合胜率。
  - `ExploiterPlayer (EP)`: 专精针对 Main Player 的弱点。
  - `MainExploiterPlayer (ME)`: 寻找并利用 Main Player 漏洞。
- **机制**：通过保存 Main Player 的历史快照形成 `HistoricalPlayers` 池。训练时，通过 Payoff Matrix（胜率矩阵）按概率选择对手（PFSP, Uniform Sampling等），确保策略能够克服过去所有版本的自己，从而实现持续的“进化”。

---

## 6. 总结

DI-star 通过其**多模态的状态空间**设计（空间+标量+实体），成功地将星际争霸II的复杂环境信息编码为深度网络可理解的表示。其**分层自回归的动作空间**将庞大的决策问题分解为六个相互关联的子任务，既保证了动作的合法性又维持了足够的表达能力。最后，其**多目标复合奖励函数**巧妙地结合了稀疏的胜负信号与稠密的模仿学习（Z奖励）及过程奖励（战斗奖励），极大地缓解了信用分配问题，并引导智能体学习人类般的宏观策略。结合分布式 V-Trace、UPGO 和 League 自我博弈，DI-star 构建了一个完整且高效的大型复杂游戏AI训练方案。
