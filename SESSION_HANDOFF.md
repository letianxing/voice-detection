# 新AI会话入口：小艾克斯系统（2026-09-16）

## 先读哪里
1. ~/Golands/voice-detection/AGENTS.md：主历史与用户约束，顶部项目摘要和最新追加优先，历史部分有已被替换的规则。
2. 要修改哪个项目，再读该项目AGENTS.md和CLAUDE.md。六项目均已分别记录功能，不再只有链接。
3. 最新架构说明：/Applications/声音/可插拔仿生注意力与论文依据.txt。
4. 验收：/Applications/声音/六项验收与准确率计算.txt；最新行为以单人语境连续性/仿生打断/多人下一版等更新说明为准。

## 当前实际状态（本轮已重新查询）
本机ROS/Memory/Vision/Voice/Attention/Brain都未运行；不要擅自开启摄像头、麦克风或播放声音。
162远端Qwen health=ok，常驻GPU0。不要被旧“所有服务关闭”的历史文字误导：它指本机，远端模型保持运行。
远端接口 http://172.16.60.162:18050/v1/chat/completions，model=qwen3.5-9b。
SSH user@172.16.60.162；密码不写入交接文档。systemd用户服务robo-qwen9.service，/home/user/robo-model。
本机一键启动默认使用远端，不再启动MacQwen；BRAIN_LLM_BASE_URL可覆盖。
大量未提交修改属于当前工作成果，不要reset/clean/revert。未创建git提交。

## 用户最重要的交互要求
- 声纹和人脸分别识别，同一Person可有多声纹模板；冲突优先可靠声纹归属话语，但不能改写不匹配的脸。
- 未对机器人说话：安静记录；对机器人发言：结合时间/说话人/背景/历史自然回答。
- 单人自言自语或多人聊天后无声转向机器人也能引起基于语境的回应，不是只支持多人。
- 机器人问完后，同一可靠说话人转头回答仍可继续，不要求每句保持凝视。
- 普通主人出现需锁定且安静5秒才问候；群体接话/明确问题优先，不问候抢话。
- 名字“小艾克斯”可无视觉/普通音量唤起；大声/咳嗽/惊跳与社交插话分级处理。
- 千问自主解释语境，不要新增“你说呢->最后一句”的场景硬编码。默认温柔短答，复杂比较总结允许完整1–2句。
- 插拔视听、记忆上下文、内部状态；当前没有内部状态也能正常工作，缺失明确标注，不造内部情绪数值。

## 最新实现（2026-09-16 第三轮：跨会话熟悉度 + 识别置信度耦合）
- 新增第五个来源 cross_session_memory：familiarity.py 后台单线程查 hri-memory-service
  （all_sessions + entity_id + kinds），统计该人在以往会话里真实交谈过的不同会话数，4 个饱和为 1.0。
  快循环只读缓存；「服务查不到」与「没有记录」是两种状态，前者不出现在结果里，不会被当成陌生人。
  时间顺序：本会话参与者约 1.4 秒越过邀请阈值，跨会话熟人约 2.3 秒，陌生人永远不会。
- 记忆通道整体乘以识别置信度（人脸/声纹取较高者），语言的「应答」脉冲同样乘。
  这是识别精度与注意力唯一的耦合点：识别退化时自动退回只靠视听与唤醒词，而不是张冠李戴。
- 角色（主人/熟人/陌生人）判断：不进入交流意愿层。注意力里起作用的是「测到的互动历史」与「识别置信度」，
  不是被赋予的头衔；角色仍保留在第一层 importance（主人先验 .5，只影响看哪里）。权限与内容仍归 Brain。
- 新增用的是 ABI v1 预留的通用键值扩展位 memory_familiarity，未改任何结构体布局，旧插件可照常加载。
- brain/gaze_invitation.py 与 tests/test_gaze_invitation.py 已按用户确认删除。
- 验证：Attention 143、Brain 62、C++ attention_plugin_test 9、DOM 通过；
  scripts/verify_attention_algorithm.py 九个离线场景全过，未开硬件。

## 上一轮实现（2026-09-16 第二轮：注意力插拔与统一算法）
把记忆上下文和当前语境文本与视听一起交给注意力统一计算，原来为"注意力只有视听"而写的下游特例被取消。
1) 输入源插拔：attention_sources.py 现有 audio_visual / memory_context / linguistic_context / cross_session_memory / internal_state。
   8092 新增"注意力输入源与算法"面板，可勾选来源、切换算法；选择写 .run/attention-config.json，重启保留。
   ATTENTION_SOURCES 仍是初始默认（audio_visual,memory_context,linguistic_context）。取消勾选=消融该通道证据，不是关硬件或安全反射。
2) 算法插拔：C ABI include/robot_attention_perception/attention_plugin_abi.h，定长 POD、启动核对 sizeof、带通用键值扩展位。
   av_memory_language_v1（默认 .so，视听+记忆+语境）与 builtin_audio_visual_v1（内置仅视听对照基线）。
   编译 bash scripts/build_attention_plugins.sh；缺 .so 不阻断启动，回退基线并在面板显示加载错误与"无交流意愿层"。
3) 算法本身两层：原有归一化竞争（看哪听哪不变）+ 新增按人累积的交流意愿对数几率。
   视听与记忆连续输入，语境每句一次脉冲（靠 event_id 只施加一次）；应答句式必须与"还在等谁回答"合取才计分。
   状态 IDLE/OBSERVING/ENGAGED/INVITED/EXPECTED_ANSWER，随每帧输出到 attention.engagement。
4) 覆盖掉的特例：两人讨论后转头看机器人、单人自言自语后看过来 -> INVITED（约1.4秒，addressed_to_robot=false，Brain据此开口）；
   机器人问完后转头回答 -> EXPECTED_ANSWER 一句放行，不要求保持视线；称呼别人 -> 负证据；陌生人长时间盯着 -> 渐近值低于阈值，永不邀请。
5) Brain：新增 attention_trigger.py 只消费 engagement；dialogue_continuity.py 改为 ExpectedAnswer，只产生"我问了谁什么"的记忆事实。
   brain/gaze_invitation.py 与 tests/test_gaze_invitation.py 已停用但未删除（未提交内容不擅自删），确认后可自行删除。
6) 记忆存储：注意力快循环只用 Brain 工作记忆快照，不做长期检索、不在50ms循环里调LLM，这是刻意的。
   pacific-rim memory_v2_service 未接入也不需要接入快循环；将来若要"跨会话熟悉度"这类先验，作为新来源走同一套来源插拔协议，不改算法 ABI。
详细见 /Applications/声音/可插拔仿生注意力与论文依据.txt（已更新为第二版，含七篇原始论文与各自适用边界）。

## 上一轮实现（第一版，仍有效的部分）
AttentionSources 首版：默认 audio_visual,memory_context，internal_state 默认关闭，接收 /brain/state 与 /brain/internal_state（schema_version1，同session、有限值、1500ms时效），无生产者时 missing，不代填心理状态。
Brain snapshot/ROS 输出 attention_memory（近期参与者、当前等待回答对象，本轮追加 question 原文）。
CompetitionAdapter 无内态时不制造 motivation；arousal 数学默认1只是无调制。
当前 MacTTS 还未换成神经情绪音色；Qwen3-TTS Serena 已核实选型但未部署。pacific-rim tts_service 缺核心 tts_engine.py 且取消/结束契约不完整，不能宣称兼容替换完成。

## 关键验证
本轮：Attention 143 项、Brain 62 项、C++ attention_plugin_test 9 项、8092 DOM 全部通过。
scripts/verify_attention_algorithm.py 用真实来源适配器 + 真实 .so + 真实融合跑九个离线场景（含关闭记忆来源的消融、跨会话熟悉度、识别不可靠三组对照），未开摄像头/麦克风/网络。
python3 scripts/mac_stack.py check 通过，已含注意力算法 .so 预检（缺失只降级不阻断）。
Voice最近73、Vision8、HRI6为较早各自相关变更的结果，不要说本轮全部重新跑过。
真实远端Qwen无播放测试已做；所有传感器现场准确率未验收。
远端模型1720token无缓存首字274ms，Brain简单样本181ms；非整体系统P95。

## 仍未完成（不能误报）
0. 交流意愿层的权重、τ 和 0.55/0.72 阈值是工程先验，没有做现场 ROC 标定；不能引用 Katzenmaier/Mallidi 等论文的准确率当作本机指标。
1. 新插件/无声邀请/转头回答的现场全栈验收、误打断与身份偏移校准。
1b. 称呼他人只能识别已登记的人名（person_id 即注册时说的名字），陌生人没有名字时该特征恒为0，不是"检查过了"。
1c. 语境特征是通用句式与词面重叠，不是语义理解，也不调用大模型；话题延续只与机器人参与过的对话比对。
1d. 跨会话熟悉度用的是「不同历史会话数」，4 个饱和为 1.0，是工程饱和点不是人类熟悉度阈值；未做现场标定。
1e. 声纹/人脸模型未更换：仍为 CampPlus(zh) + YuNet/SFace。已完成选型调研（见下方"识别模型选型"），换模型会作废现有声纹与人脸模板，需重新登记，等用户决定。
2. 任意人数逐字实时重叠分离；现multi仅完成语音段、最多4登记参考/12秒、独立身份验证。
3. 完整人际听话人/代词指向模型、完整麦格克音素融合。
4. 情绪神经TTS实际部署、流式首音频/取消/AEC兼容。现Tingting的温柔文本不等于情绪音色。
5. 额外ROI感知资源调度和真实反馈生产者、实体转头、真实3D、微振动/可靠远场吸气。
6. VAP中文多人校准，声纹/人脸ROC与阈值现场标注。

## 识别模型选型（已调研，未更换，等用户决定）
本机实测（M4 Pro，onnxruntime 1.30 CPU EP，纯模型推理未开硬件）：
CampPlus zh 对 3 秒语音 9.56ms/次、192 维；SFace 4.02ms/次、128 维。延迟余量很大，换更强的模型在算力上完全可行。
声纹（当前 CampPlus / CAM++，3D-Speaker zh 16k）
- 3D-Speaker 官方表：CAM++ 7.2M 参数，CNCeleb EER 6.78%；ERes2NetV2 17.8M，CNCeleb 6.14%，3D-Speaker 集 6.52%。
  短句上差距更明显：ERes2NetV2 在 VoxCeleb1-O 3 秒 0.98%、2 秒 1.48%。我们大量是短句，这条最相关。
- WeSpeaker cnceleb_resnet34_LM 直接提供 onnx（26.5MB），CNCeleb 6.49%；ResNet221 5.66% 但更重。
- 建议：升级到 ERes2NetV2（中文 + 短句都更好，预估 ~25ms 仍远低于预算）。
  代价：嵌入空间变了，现有声纹模板全部作废，必须重新登记；阈值 .48/.6 要重新标定。
声纹阈值现状：0.48 匹配 / 0.6 可靠，是工程初值，没做过 ROC。换不换模型都应该先用真人数据标一次，这比换模型收益更确定。
人脸（当前 YuNet 检测 + SFace 识别）
- SFace 论文对比的是损失函数（CASIA-WebFace 训练），不能与 WebFace600K 训练的 w600k_r50 比。
  InsightFace buffalo_l(w600k_r50) IJB-C TAR@FAR=1e-4 约 95–96%，R100 约 96–97.5%，明显强于当前配置。
- 但许可是硬约束：InsightFace 代码 MIT，权重与训练数据仅限非商业研究用途；
  OpenCV Zoo 的 SFace 权重是 Apache-2.0。如果这套系统要商用，换 buffalo_l 会带来许可问题。
- 建议：先确认用途。研究/内部用可换 buffalo_l；要商用则保留 SFace，改为提高人脸质量门槛
  （更大输入、正脸筛选、多帧模板）而不是换权重。
换模型的连锁影响：现有 3 个身份档案没有参考 wav，无法用旧向量迁移，必须真人重新登记；
本轮已把注意力的记忆通道挂到识别置信度上，识别一变强，记忆通道起作用会更早更稳，这是可直接观察的收益。

## 下一步工作方式
先问清本轮用户具体目标（若已指定就直接继续），不要把所有未完成科研能力都宣称即将完整解决。
修改前先查状态、停本机；不要误停远端Qwen。保护现有数据events.jsonl、outbox.sqlite3、身份模板，不删记录伪造成功率。
Python：~/Golands/robot-attention-perception/.venv-mac/bin/python。
在相应项目目录运行 python -m unittest discover -s tests；Attention另node tests/test_live_app.cjs；HRI cargo test。
当前用户要换session，本轮只做交接，不启动现场测试。
