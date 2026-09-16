# voice-detection：当前实现与交接（2026-09-16）

## 工作规则与实际部署
- 用户授权跨六项目协作；大量未提交实现不可擅自清理或回滚。不创建子代理，除非用户明确要求。
- 修改前检查并停止本机受管服务，不擅自打开摄像头/麦克风或播放。远端162的Qwen为常驻服务，不能随本机stop_all误停。
- 机器人名称“小艾克斯”。用户要求自然多人对话、旁人记忆、视听身份、注意力约束打断、主动搭话。声纹/视觉冲突优先可靠声纹归属发言，不据此改写冲突人脸。
- 本机启动：cd ~/Golands/biomimetic-brain-test && bash scripts/start_all.sh；停止：bash scripts/stop_all.sh；状态：python3 scripts/mac_stack.py status。
- Python：~/Golands/robot-attention-perception/.venv-mac/bin/python。MediaPipe固定0.10.32，不盲目升级。
- 端口：Vision8080、Voice8090、Attention8092、Brain8094、Memory8788、ROS9090；慢脑http://172.16.60.162:18050/v1/chat/completions，model=qwen3.5-9b。
- 所有基准和单元测试仅说明对应样本/程序行为，不能描述为现场识别准确率达标。
- 不在代码/文档保存SSH密码。完整历史见voice-detection/AGENTS.md下方历史记录；本节当前状态优先于过期历史。
- /Applications/声音/Mac全系统启动与三项验收.txt保持“一键启动／分别启动”两块，详细验收独立保存。

## 本项目已实现
- 原始多通道采集、VAD、降噪、GCC-PHAT TDOA、波束形成与AEC；raw通道不足拒绝虚假方向，ILD仅诊断。Sipeed连接状态需要实际设备查询，不能凭选择profile认定连接。
- Sherpa常驻子进程流式ASR，句末处理/声纹计算移出音频回调；Whisper/Vosk备选；有界队列与模型就绪状态。
- CampPlus中文声纹。旧单模板兼容、多模板池最多8份、保留原始anchor，best/runner margin检查，原子写盘；同期高质量人脸与声纹一致才增加样本。不能凭一次弱匹配自学习。
- 陌生人满足质量门槛自动建立stranger编号，正式注册别名归一，稳定身份记录去重。正式登记两段声纹一致性检查；每身份参考wav支持从新登记/可靠后续样本补齐。
- SpeakerBeam/REAL-TSE单目标提取；multi_speaker.py完成语音段后逐参考分离、身份验证、ASR，--tse multi显式实验模式，最多4份登记参考/段长12s，同段多人的utterance_id不同。缺参考或验证失败不强行归人；mixed_fallback只记忆不放行。
- PlaybackEngine常驻音频流，Brain传PCM16。实际播放参考、回声文本/时间匹配、旧ID隔离。半余弦淡出：普通20ms、礼让40ms、停止10ms、惊跳5ms，非微秒保证。
- BioAcoustics：相对背景突增、上升沿、谱通量、2–4kHz比例、30–150Hz调制、F0Hz/voicing/斜率/停顿。不是诊断焦虑或危险。
- AST声音事件/音乐、十类genre、BPM与节拍；Cough/Sneeze分类。一般突变先定向分类，极强声保持快速反射；语义alarm有窗口延迟。咳嗽分类去重，提供Brain关怀事件。
- VAP官方权重，4秒AEC残差+实际TTS参考输入，MPS优先、独立有界worker；仅辅助轮次确认，未校准中文多人远场。
- ROS话题发布、实时8090接口、模型耗时/健康信息。声纹池/API与分离数据可供Attention使用。

## 关键文件、验证及限制
- voice_detection/{dashboard,speaker_embedding,multi_speaker,playback,persistent_asr,scene_runtime,bio_acoustics,turn_prediction}.py。
- scripts/setup_vap.py固定权重SHA；verify_bio_integration.py不采集硬件的真实模型联调；transcribe_overlap.py离线分离转写。
- 最近完整Voice73测试通过（9月15）；VAP/AST离线联调ready、无丢帧、正常退出。REAL-TSE5秒样本冷8.37s/热1.64s每目标，不是现场准确率。
- 未完成任意人数重叠逐字实时识别、F0目标增益、可靠远场吸气/情绪诊断。旧3个身份最初无参考wav，必须实际补样本，不能从embedding恢复音频。

<!-- CURRENT_PROJECT_SUMMARY_END -->

# 历史记录（上方最新状态优先）
# 小艾克斯仿生交互系统：任务交接

更新时间：2026-09-15。新会话先读本文件，勿把单元测试通过描述为现场准确率达标。

## 用户目标与工作规则
- 六项目：voice-detection、vision-detection、robot-attention-perception、hri-memory-service、biomimetic-brain-test、robo-model；均在 ~/Golands。
- 用户要日常视听注意力、身份/记忆归属、实时S2S、注意力约束的打断、主人主动搭话。机器人叫“小艾克斯”。
- 用户要求修改前停服务。目前全部服务已停，不要擅自启动摄像头/麦克风或自动播放。
- 不清理或回滚现有未提交更改；此前大量实现未提交。不得声称全部功能已经完成。
- 不创建子代理，除非用户明确要求。
- 启动说明 /Applications/声音/Mac全系统启动与三项验收.txt 只保留“一键启动”“分别启动”两大部分；详细验收另存。

## 启动和环境
- cd ~/Golands/biomimetic-brain-test && bash scripts/start_all.sh
- 停止 bash ~/Golands/biomimetic-brain-test/scripts/stop_all.sh
- 状态 python3 ~/Golands/biomimetic-brain-test/scripts/mac_stack.py status
- 默认会话 mac-conversation-test；总览8092，视觉8080、语音8090、Brain8094、Memory8788、ROS9090、Qwen18050。
- Python ~/Golands/robot-attention-perception/.venv-mac/bin/python。
- MediaPipe固定0.10.32；此前1.0.1有native crash，勿盲目升级。
- 模型已有Qwen3.5-9B Q4_K_M、Sherpa/CampPlus、AST、genre、YOLO/YuNet/SFace/FER+、Hand/Face Landmarker、REAL-TSE、VAP。

## 已实现
### 身份、记忆
- 用户面对镜头说“你好小艾克斯，我叫小天，请记住我，我是你的主人”自动注册；不需要页面操作。普通熟人去掉主人声明。
- CampPlus两段声纹一致性预检(.6)，SFace连续5帧+2帧唇动+模板一致性(.7)预检；先预检两边再提交，部分失败有提示，非跨服务事务。
- 陌生声纹满足时长/清晰/非回声条件建立stranger编号；正式登记保留别名。原始匹配阈值声纹.48、人脸.38，未进行现场ROC校准。
- 声纹与视觉冲突优先可靠声纹归属发言；标记identity_conflict和visual_person_id，不改写冲突人脸。已移除“唯一可见人就用声音给脸改名”的捷径。
- 稳定身份档案去重；每句话独立记忆。HRI追加JSONL，Brain sqlite outbox。跨会话all_sessions只允许同subject/robot检索。
- 当前问题、个人/别名历史、机器人/环境历史、最近2分钟最多8句和证据交给Qwen；不让历史盖过当前问题。
- 8092历史分页、按会话/人/关键词过滤、实际模型上下文、事件追溯；无原始音视频录制。

### 注意力与主动性
- InteractionFusion 是日常交互权威策略，文件 attention/robot_attention_perception/interaction_fusion.py；上游ROS/local只是候选，不可凭旧listen直接放行。
- 状态 IDLE/ACQUIRING/VISUAL_FOCUS/AUDIO_FOCUS/LISTENING/RESPONDING/ORIENTING/AMBIENT_SPEECH。
- 所有人可用持续注视吸引注意力；仅视觉关注不代表正在对机器人说话。
- 普通音量、无人入镜也可直接叫名；只叫名答“我在，你说”；句中提及不当呼叫。6秒听觉关注窗口，未知方位不编角度。
- 同时效声源、注视、唇动、空间证据放行发言；finalize_gate按同期历史决定最终句；可信声纹可辅助已关注者最终句。
- 主人必须锁定后连续安静5秒才主动（BRAIN_PROACTIVE_WAIT_MS）。说话重置等待；“你说是吧”优先接续旁人对话，不问候；已交谈当次在场不重复问候。
- 最新修复 owner/none跳变：visual_focus.py进入阈值比保持严格；进入800ms、保持gaze>=.5/face>=.65，短丢失最多650ms保留目标但listen=false；明确转开持续300ms释放。语音结束adopt目标，不重新等800ms。
- Brain允许已生成主动回复在短hold中保留；不在hold时新启动问候。Person保留owner角色。
- .run/attention-decisions.jsonl 每轮数值日志，异步队列，10MB*3轮换；无图像/声音/声纹向量。此前旧记忆person_observation约1Hz，不足以定量回放50ms跳变，不能声称现场抖动已经减少多少。

### 声音、打断与最新仿生补充
- Sherpa常驻进程流式ASR，Qwen SSE短句重叠TTS；Voice持久OutputStream提供实际PCM回声参考。
- 自回声归属拒绝、旧生成取消、陈旧播放回调隔离，注意力确认打断。
- bio_acoustics.py 新增稳健背景z分数、上升沿、谱通量、2–4kHz比例、30–150Hz包络调制粗糙度、F0 Hz/voicing/slope/pause。不是焦虑/危险诊断。
- VAP官方模型vendor/voice_activity_projection，MIT，commit及权重SHA在PROVENANCE.md；setup_vap.py固定版本校验下载。静态CPC config避免运行时联网。独立子进程、队列上限1、每200ms输入4秒AEC麦克风+实际TTS参考，MPS优先，失效时不放权。
- VAP 真实样本离线worker约34ms，bio特征约.25ms/20ms帧（样本测量，非全栈保证）。VAP未校准中文/多人远场，辅助调整确认窗，不独立允许打断。
- turn_taking.py：附和“嗯/对/好的”继续；明确停止/紧急词快停；普通已放行插话在短句边界让话，上限350ms；名字120ms确认，普通120或200ms。
- playback.py半余弦淡出：普通20ms，礼让40ms，明确停止10ms，惊跳5ms；取消期间参考使用实际淡出PCM。不是微秒级。重复stop不重置fade；旧id不能停新播放。
- 惊跳：原声音阈值-12dBFS、突增24dB、预热2.5s、冷却8s；小瞬态仅显著度定向。AST显著突变后用2秒窗口检查Shatter/Explosion/Gunshot/Screaming，高分且新鲜才发semantic_alarm_confirmed；有分类延迟，非安全报警器。音乐不再一概屏蔽符合证据的异常。
- 声源GCC-PHAT TDOA；无原始通道拒绝虚假60度，增加ILD诊断（未校准，不参与角度）。混音只用raw阵列通道，避免硬件beam/ref重复污染。
- confirmed惊跳优先，新问题暂存后处理；普通旁人不打断播放，未播问候可延后（独立计数）。

### 视觉和实验
- FER+已修复输入0–255尺度与置信过滤；唇动用MediaPipe Face几何和时序，不用像素差；invalid标未知。新增Face模型已下载/启动预检。
- 人脸框曾写死stranger，已修复。
- 惊跳使用固定脸框，不混人体框；物体增长和中心大物体突然出现需确认。
- 双闪：真实PCM窄带音脉冲+图像亮度脉冲时间绑定。麦格克：仅双唇音闭唇几何冲突，非完整读唇/音素融合。

## 当前验证结果
最近完整：Voice68 tests、Attention85、Brain44、Vision8通过；HRI此前6通过。页面DOM与node语法通过。新增测试文件见各tests。
- 已测试：淡出波形/实际参考/重复停止，F0/粗糙度/瞬态，AST语义事件去重/过期，VAP不独立放行、短句让话、目标抖动/释放、主动短hold。
- tests/test_daily_interactions.py 跨Attention->Conversation->Brain->mock TTS完整事件序列，非真实硬件试验。
- /tmp/check_bio_runtime.py 可离线运行VAP和SceneRuntime；依赖/tmp/x-vap-inspect中的官方示例wav，换机器需改为持久测试样本。
- /tmp/x-vap-inspect 是研究检出的官方仓库；运行只需本项目vendor与weights，不应依赖/tmp。

## 尚未完成 / 必须如实告知
1. 最新VAP/淡出/保持阈值还未现场全栈验收（遵守停服务要求）。不要声称现场打断成功率/身份准确率已达标。
2. 本轮新能力还未完整同步到 /Applications/声音 验收说明和项目README。下一步必须更新：仿生能力、VAP启停/预热状态、时间参数、保持状态含义、限制与论文来源。
3. 需要补一次真实但离线SceneRuntime整合多进程退出/输入队列验证，已有单独worker验证。
4. 确认setup/启动预检覆盖VAP依赖einops、cpc_config、model.pt（pyproject scene已加einops，mac_stack_config已加文件）；未把VAP ready设为启动硬阻断，UI会显示模型加载状态。
5. 稳健远场吸气检测、准确心理焦虑判断不支持；没有微振动传感器，不能测微震。普通RGB不是真3D，不执行实体转头。
6. 多人重叠逐人完整ASR/声纹分离未完成；SpeakerBeam是已有可选提取，没实现逐人并行全环境转写。F0定向增益/新流式分离不能假装完成。
7. 完整麦格克音素融合未实现；现有双唇音检查只是有限实验。
8. 无法控制外部播放器（Apple Music等）；能停止的只有本系统Voice PCM。未实现任意外部音乐自动暂停。
9. 中文VAP标注/适配、阈值ROC校准、系统级误打断/延迟P95尚缺真实数据。
10. 粗糙度是特征而非完整标准心理声学粗糙度模型；2–4kHz不单独触发惊跳；语义事件存在AST窗口和调度延迟。
11. 当前社会策略的附和列表和紧急词是工程基线；复杂讽刺/语义相关性不是完备分类器。
12. 检查长期性能：VAP重复4秒推理MPS与Qwen共用GPU，尚未全系统负载实测。

## 下一步建议顺序
- 核对当前git diff；优先完成新能力文档、启动依赖、离线联调与回归。
- 用户再次明确要现场测试/启动时再开服务，观察8092目标hold、VAP ready、bio、播放状态和decision日志。
- 用“六项验收与准确率计算.txt”执行A/B/C人物与注意力场景；统计目标振荡/误打断/身份误绑，不做一次成功的准确率承诺。
- 不删除旧记忆以伪造去重成功；旧事件保留，必要时采用新会话作验收并显式考虑跨会话召回。

## 研究依据（已阅读官方/原始研究）
- https://www.isca-archive.org/interspeech_2007/kalinli07_interspeech.html 听觉显著度
- https://pubmed.ncbi.nlm.nih.gov/26190070/ Arnal2015尖叫粗糙度；勿等同2–4kHz载频
- https://www.isca-archive.org/interspeech_2022/ekstedt22_interspeech.html VAP
- https://aclanthology.org/2022.sigdial-1.51/ 韵律与轮次
- https://github.com/ErikEkstedt/VoiceActivityProjection 官方代码/权重
- https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.575566/full 呼吸研究（呼吸带/近讲采集，不可推导远场吸气可靠）
- https://arxiv.org/abs/2401.14717 声学+LLM轮次/附和
- https://www.isca-archive.org/interspeech_2019/li19m_interspeech.html Direction-Aware SpeakerBeam

## 续会话收尾（2026-09-15）
- 已完成本AGENTS主记录与其余五项目入口。
- 已新增 /Applications/声音/仿生打断与注意力保持更新.txt，复制到Brain docs；启动/六项验收/README同步了VAP和最新参数，旧文档时间参数以新更新说明为准。
- scripts/verify_bio_integration.py 持久化了无需摄像头/麦克风的真实模型联调。实测AST ready、VAP ready/valid、bio valid、队列dropped_frames=0，VAP样本30.65ms，全部进程关闭正常。只验证离线合成音频，不代表真人/中文准确率。
- 上述未完成列表第2、第3项已完成；第4项依赖/文件已核对。最新未改变算法代码，因此不重复全套测试。
- 收尾发现用户另处再次启动all；已按既有“修改前停止服务”指令执行stop_all.sh。新会话应重新查询状态。
- 仍需现场验证、中文VAP/阈值校准、长时间性能；其余明确未实现项仍适用，不应冒称硬件或算法无法支持的能力已补齐。

## TTS延迟排查（最新）
- 用户说TTS慢且Sipeed未连接。读取活跃Brain与Qwen日志：句末到Brain约542ms、首段到播放约524ms，首字6110ms；Qwen prompt eval 5869ms/1749tokens（另一次6224ms/1830tokens）。主瓶颈是长prompt prefill，非say本身。
- 新增 brain/prompt_context.py：简化系统词、删除重复门控字段/调试浮点与空感知，recent/history按人+文字+时间去重，保留12条候选历史和8条recent上限、身份冲突。明确当前问题已通过门控，避免模型又质疑是否对它说话。
- llm.py使用build_context；runtime增加memory_started_ms/memory_finished_ms/llm_request_ms；8092显示记忆检索、模型首字。
- Brain46测试、DOM通过；尚未重启模型实测优化后首字延迟，不许把字符减少当实测速率。保存样本user context序列化诊断近似3748->933字符（旧统计含trace字段，不是精确旧HTTP请求token）。
- 最新系统设备查询：仅MacBook内置麦克风/扬声器，CoreAudio无MicArray，USB无明确Sipeed名称。需要用户检查线/接口，不能软件修复物理断连。
- 已停止all并确认所有端口未运行。TTS必要记忆不删除、原始证据保留；跨人重复文本不会被去重。

## 极短温柔回复（最新用户要求）
- 用户要求提示词TTS温柔可爱、逻辑通畅、安抚、3～9字。prompt_context.py已添加一句正文3～9字（标点不计），普通和主动模式一致，事实准确不盲目附和；llm默认max_tokens192降48。
- 这是提示词约束，尚未真人/真实模型验证每次字数达标；固定注册/快脑提示仍可能超过9字。不能说已经全链路硬限制。不能简单截断事实句来满足字数。
- Qwen脚本核实默认：llama-server GGUF Q4_K_M、GPU layers99、Flash Attention on、ctx4096、parallel1、batch512/ubatch256、8threads；Brain已SSE stream=true、enable_thinking=false。部署支持流式，并非当前在用非流式。
- 本轮未改部署后端或盲调batch；已有日志瓶颈为prompt prefill，真实优化后TTFT尚未复测。服务保持停止。

## Qwen实际性能核查（已完成）
Qwen速度核查与优化 2026-09-15
实际模型：Qwen3.5-9B Q4_K_M，llama.cpp 89fe242，Apple M4 Pro。
验证：Release构建、Metal启用、33/33层GPU卸载、Flash Attention启用。不是CPU退化。
模型独立运行测试，未启动摄像头/麦克风/TTS。
865 token无缓存同输入：batch512/ubatch256首字2692ms；2048/512约2641ms；2048/1024约2619ms，改善约3%。
1720 token：首次首字5188ms；完全相同输入缓存命中109–110ms。缓存命中不能代表新问题延迟。
精简真实上下文测试：你好首字793ms/完成959ms；保存的天气问题首字1222ms/完成1469ms；同请求缓存重复105ms。
已改默认batch2048/ubatch1024；用户环境变量仍可覆盖。将history置前、当前问题置后，尝试保留公共前缀。内容变化或历史排序改变时仍可能大量重算。
真实日志说明prefill约330token/s，CPU/GPU共用不是主要原因。未替换模型，不保证所有新问题低于1秒。
测试中3–9字提示词仍有超长和天气推断问题；不得声称字数与内容达标。须另行完善输出约束，不能靠粗暴截断保证。
本次未改变TTS音色或播放速度；实际出声仍包含句末检测、记忆、合成/播放约0.5秒等环节。

基准JSON在Brain docs/benchmarks；可重复脚本scripts/benchmark_qwen.py（仅模型18051端口，开始前确认端口空闲）。测试服务均退出。

## 最近话题承接与咳嗽修正
- 实际18:07对话：天气问题已在recent，但“你说呢”仍回答自我介绍。prompt_context新增current_topic（最近非空非附和句），承接问句明确优先该主题，不受to_robot=false或不同身份标签阻挡；不捏造天气。
- AST event_scores增加Cough/Sneeze；强咳嗽分类高于弱alarm时不惊跳；cough事件30秒去重。普通阈值触发且raw<-6dBFS先orient待分类，极强声仍即时惊跳。-6是工程初值未现场校准，不是人类生理常数。
- Brain确认cough后等待至少1秒安静、有效人物注意力且无播放/生成才说“慢慢来，别着急。”；6秒过期，不抢普通发言，不诊断身份/健康。现不精确定位咳嗽者，仅环境关怀。
- 按既有指令停止all；新改动待重启现场验证。

## 结构化语境泛化（最新用户纠正）
- 用户不接受为“你说呢”等列场景规则。已删除正则current_topic/强选最后句；千问根据recent/history时间、说话人、to_robot、背景、身份冲突自主决定延续/换题。
- runtime recent含utterance_id/session/start/end/person/identity/background；prompt_context压缩视觉为人物/角色/有效表情/注视/唇动语义，不传完整低层检测字段。history同样提取背景；保留不同人/不同时间的同文字。
- 每轮current_turn.model_context记录实际提交的结构化输入，可从8092原始快照回溯。仍最多recent8句/2分钟、history12条，是预算内的召回子集，不宣称全部记忆全量提交。3–9字要求保持。
- 用户背景中的身份分裂未自动强合并；身份冲突继续显式告知模型。旧天气错误不能由删记忆掩盖。

## 飞书PRD连续竞争融合（最新）
- 阅读用户指定注意力PRD §3.2/3.4，新增attention_competition.py和competition_adapter.py，InteractionFusion每轮运行，visual_focus用连续焦点裁决竞争切换。每路视觉/听觉预算独立，dt指数更新、H/C/R、可靠关联增益、先验/目标/统计偏差已接。
- 图像对象由reflex_attention.object_tracks供候选，非安全/对话许可。输出attention_distribution至ROS brain_input/日志/UI。
- feedback/confirmed goals/extra allocation接口有实现和测试，但生产驱动力/任务接线、额外感知执行器和真实反馈尚未接；allocation显式executable=false。不能声称全部PRD已融合。
- Attention92项及DOM通过，Brain回归见后续结果。详细已做/未做记录：/Applications/声音/飞书注意力方案融合记录.txt及attention docs同名文件。

## 当前会话目标接线（2026-09-15）
- 必要并已实现：Brain snapshot携带session_id/stamp_ms；既有/brain/state每0.5s发布，Attention订阅；session_attention.py把当前thinking/synthesizing/speaking且确有busy状态的同会话对话转换为已确认视觉目标支持。
- 1200ms过期自动撤销；完成/取消/惊跳不继续支持；人脸不在场或声纹人脸冲突不赋错误人脸目标。该支持不创建对话许可。
- 暂不启用额外感知调度：目前基础模型已持续运行，缺少独立额外处理预算/执行反馈，重复调用会增加延迟。observe_result接口保留，但不伪造LLM回答等于取得事实。
- 驱动力系统没有当前已接入的生产者，不能合成虚假生理值。当前任务采用真实对话任务；未来其他任务同样需显式来源/时间/取消协议。
- 实体转头无硬件执行端，维持未执行。
- Attention94项通过；Brain回归49项通过（完成后已核对）。未重启全栈，跨进程现场仍待验收。

## 多人群体对话下一版（最新）
- 用户明确不限两人，重点记忆语境自然接续。新增attention group_conversation.py，ConversationObserver逐句标记群体成员/听话人/重叠时间。明确机器人邀请开启120s群体参与；明确群体问句可放行，向人称呼拒绝机器人门控；没有让所有群聊发言自动回复。
- Brain pending_group_turn留700ms安静窗口；真人随后发言撤回待答；不抢正在TTS。最近语境16句/2分钟，保留多人独立观点。默认3–9字放宽为比较总结允许1–2句，LLM预算128tokens。
- Attention97测试通过、Brain50全套通过，新增group等待测试单独20项通过。文档 /Applications/声音/多人对话下一版.txt与Brain docs。
- 这仅交互层下一版原型，不是多人重叠转写已实现。SpeakerBeam当前只单目标整句，VAP仍人群一路+机器人一路。完整N方轮次模型、多人声学分离需继续研究实现。不要称已无缝完成。

## 远端4090部署与声纹池（2026-09-15，最新）
用户明确授权SSH到172.16.60.162/user，GPU0部署Qwen优先，可清理占用。密码不要记录或输出到文档。
- 已git clone ~/robo-model（HEAD b2063ca）；从本机传输llama源到third_party/llama-cuda，CUDA12.8 Release sm89构建。编译初期Mac AppleDouble文件误入，已删除本次传输的._文件后构建成功。
- 权重Qwen3.5-9B Q4_K_M SHA cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13，远端本地一致。
- systemd用户服务robo-qwen9.service已active/enabled，Linger=yes，GPU0 CUDA_VISIBLE_DEVICES=0，LAN bind172.16.60.162:18050。
- GPU0训练PID2780260占89%算力，TERM无效后按授权KILL；该进程也占GPU1少量资源，其他GPU1服务未动。远端模型现在运行中，不要随本机stop_all停掉。
- 实测1720token无缓存首字274.4ms、prompt230.6ms约7459tok/s；Brain客户端181ms首字。不要把它当全系统P95保证。
- 本机scripts/llm_endpoint.py默认远端，mac_stack all跳过本机Qwen，检查远端健康；run_brain.sh也设远端默认。env BRAIN_LLM_BASE_URL可覆盖。本机未启动摄像头/麦克风/Brain，只测试真实客户端。
- speaker_embedding.py已多模板池、anchor防漂移、最大8、跨人margin .06、自动增长需要同期人脸>=.8和高质量声纹>=.7且margin .12。旧单模板兼容，原子保存。voice-pool接口由Attention单线程任务送去，避免融合循环等待。
- 新正式注册保存每Person参考wav；multi_speaker.py和--tse multi只支持已登记参考的整段多路提取+身份验证+ASR，预算<=4人/12s；失败保存mixed_fallback且不放行回复。同区间不同人utterance_id不同。不是逐字实时全重叠解决。
- 实际现有3档案没有参考音频，无法凭向量恢复；需真实重新登记/补参考。离线提取5秒音频模型冷加载8.37s、热1.64s每路。无真人准确率标签。
- 完整人际addressee未实现；不要声称所有需求已全部完成。文档 /Applications/声音/162慢脑与声纹池部署记录.txt。
- 最终补充验证：Voice73测试通过（更新pool rank测试桩）、Attention97、Brain52通过；新增distinct separated IDs用例通过。SSH已退出后health仍ok，远端常驻已验证。
- 自动高置信视听pool更新成功且缺参考时，会从仍对应同一utterance的缓存音频保存每人参考；旧参考缺失不再要求页面操作，但需后续合格样本，不能人工制造。
- 仍未解决完整人际语义对象模型和任意人数逐字重叠分离；当前multi为显式完成段实验路径，不应宣传全完成。

## 转向接话与tts_service评估（2026-09-16）
- 新增Brain gaze_invitation.py：最近90s同episode>=2说话人，目标是参与者或当时可见者，VISUAL_FOCUS持续1.5s且安静1s，无busy/pending后触发非语言邀请。空user_text+attention_trigger送Qwen，不伪造ASR。同目标/episode去重、直接问话优先，取消待发无声回应，不补发主人问候。
- UI显示gaze_invitation_started/无声输入。模型请求保持结构化近期内容，千问自由承接，不强制特定话题答案。
- tts_service只读检查：node.py依赖的tts/adapters/tts_engine.py缺失；AudioFrame仅PCM与header，不含显式轮次/结束/取消。现不能直接替换。未修改pacific-rim任何IDL或生成代码。
- 兼容方向：tts_service做合成，协议桥输出到Voice播放器/AEC，Brain继续拥有取消；需要正式request/sequence/end/cancel契约，用Dashboard或./pr data-format更改，非手工写IDL。
- 详细 /Applications/声音/转向接话与TTS服务兼容评估.txt（Brain docs也有）。本机服务关闭，远端Qwen仍运行。
- 真实远端非播放样本2条通过合理性核对：问带什么答饼干，无声转向答参与确认；不是现场准确率。

## 可插拔注意力输入（2026-09-16）
Attention现在有视听/记忆/可选内部状态支持源；本项目既有感知、记忆存储或模型服务职责不变，不能把“来源关闭”误作停硬件/安全通路。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。没有内部状态也可试用第一版，但现场可靠性未验收，不能搬用论文准确率。

## 会话切换交接（2026-09-16，最新）
新AI先读 `/Users/letianxing/Golands/voice-detection/SESSION_HANDOFF.md`，再读本项目当前摘要。
本轮已核查：本机全部感知/Brain服务停止，162远端Qwen健康运行。最新Attention101/Brain60/DOM测试通过，不等于现场准确率。没有提交git，保护所有未提交修改和用户数据。

## 注意力统一为 视听+记忆+语境，并做成可插拔（2026-09-16，最新）
用户指出：之前为"注意力只基于视听融合"而写的特殊 case（注意力未触发时两人对话、其中一人看向机器人才调上下文；
自言自语后由上下文决定是否回话），本质上是因为注意力没有拿到记忆上下文和当前环境 ASR 文本。正确做法是把记忆、
语境与视听一起交给注意力综合计算，并把视听/记忆/（将来的）内部状态对注意力做成插拔式，算法本身也做成可切换的 .so。
本轮按此实现，特例代码不再需要。
- 来源插拔：attention_sources.py 增加 linguistic_context；四个来源可在 8092 勾选，写 .run/attention-config.json，重启保留。
  关闭=消融该通道证据，不是关硬件或安全反射；internal_state 仍默认关闭且无生产者，缺失标 missing，不造心理数值。
- 算法插拔：C ABI attention_plugin_abi.h，定长 POD + 启动 sizeof 校验 + 通用键值扩展位；ctypes 加载。
  av_memory_language_v1（默认 .so）与 builtin_audio_visual_v1（内置仅视听对照基线）。
  bash scripts/build_attention_plugins.sh 编译；缺 .so 回退基线并在面板报错，不阻断启动。
- 算法：第一层保持原归一化竞争（看哪听哪不变）；第二层新增按人累积的交流意愿对数几率，
  视听/记忆连续输入、语境每句一次脉冲，应答句式必须与"还在等谁回答"合取才计分，称呼他人为负证据。
  阈值 0.55 观察 / 0.72 交流；状态 IDLE/OBSERVING/ENGAGED/INVITED/EXPECTED_ANSWER，随帧输出 attention.engagement。
- 下游：Attention 内部 INVITED -> interaction_phase=INVITED（listen=true、addressed_to_robot=false），
  EXPECTED_ANSWER -> LISTENING（addressed_to_robot=true），finalize_gate 逻辑不变。
  Brain 新增 attention_trigger.py 只消费 engagement；dialogue_continuity.py 改为 ExpectedAnswer 只产生记忆事实，
  放行后仍由 Brain 把自己问过的原句附给千问。brain/gaze_invitation.py 及其测试已停用未删除，待用户确认再删。
- 记忆：注意力快循环只读 Brain 工作记忆快照，不做长期检索也不在 50ms 循环调 LLM。
  已按用户提示查看 pacific-rim/module/service/memory_v2_service：能力更强（DuckDB 事实源、实体图、向量召回），
  但当前注意力机制不需要它，也不应把它放进快循环；将来要"跨会话熟悉度"这类先验，作为新来源接入，不改算法 ABI。
- 验证：Attention 132、Brain 66、C++ attention_plugin_test 8、8092 DOM 通过；
  scripts/verify_attention_algorithm.py 真实来源+真实 .so+真实融合跑六个离线场景含消融对照，未开硬件。
  权重/阈值是工程先验，未做现场 ROC 标定，不能把论文或单测结果说成现场准确率。
- 本轮未启动任何采集，未改远端 162 Qwen，未 git 提交，未删除任何未提交文件。
- 论文与协议细节：/Applications/声音/可插拔仿生注意力与论文依据.txt（第二版）。

## 跨会话熟悉度、识别置信度耦合与身份角色定位（2026-09-16，最新）
- 新增第五个来源 cross_session_memory（familiarity.py）：后台单线程查 hri-memory-service
  （all_sessions + entity_id + kinds=heard_utterance/dialogue_turn），统计该人在以往会话里真实交谈过的
  不同会话数，4 个饱和为 1.0。快循环只读缓存，不阻塞 20Hz。
  「服务查不到」与「没有记录」是两种状态，前者根本不进结果，不会被当成陌生人。
  时序刻意分层：本会话参与者约 1.4 秒越过邀请阈值，跨会话熟人约 2.3 秒，陌生人永远不会。
- 识别置信度耦合：记忆通道整体乘以识别置信度（人脸/声纹取较高者），语言的「应答」脉冲同样乘。
  这是识别精度与注意力唯一的耦合点。识别退化时系统自动退回只靠视听与唤醒词，不会张冠李戴。
  相应地，记忆类 reason 位也只在识别足够时才上报，日志不会声称用了被缩放掉的证据。
- 身份角色（主人/熟人/陌生人）的判断：不进入交流意愿层。
  理由是角色是被赋予的头衔，属于权限与内容问题（Brain 负责）；注意力该用的是可测量的东西——
  实际互动历史（熟悉度）与识别置信度。用角色会带来错误偏置：主人的无意一瞥被过度响应，
  客人真心提问反而被压低。角色仍保留在第一层 importance（主人先验 .5），只影响「看哪里」。
- 实现用的是 ABI v1 预留的通用键值扩展位 memory_familiarity，未改任何结构体布局；旧插件可照常加载并忽略。
- brain/gaze_invitation.py 与 tests/test_gaze_invitation.py 已按用户确认删除，运行时不再有任何场景分支。
- 识别模型选型已调研（实测 CampPlus 9.56ms、SFace 4.02ms，余量很大），未更换。
  声纹建议 ERes2NetV2（中文与短句更好）；人脸 buffalo_l 更准但权重仅限非商业研究，需先确认用途。
  换模型会作废现有声纹/人脸模板，必须真人重新登记。详见 SESSION_HANDOFF.md「识别模型选型」。
- 验证：Attention 143、Brain 62、C++ attention_plugin_test 9、DOM 通过；
  verify_attention_algorithm.py 九个离线场景（含三组消融/对照）全过，未开硬件。
- 本轮仍未启动采集，未改远端 162 Qwen。

## 统一订阅点、v2 注意力算法与识别模型替换（2026-09-16，最新）
- Global Workspace：robot_attention_perception/global_workspace.py 是感知循环的广播步骤，只发布竞争结果，
  不做决策。内容含 cycle/focus/target/engagement/coalition/sources/algorithm/provenance。
  入口 GET /api/workspace、?since=N 回放、/api/workspace/stream（SSE）、ROS /attention/workspace。
  订阅者落后丢帧并计数，不阻塞感知循环；Brain 现有 /attention/brain_input 通路未改动。
  这是 LIDA 认知周期里的广播一步，不声称实现了意识或完整认知周期。
- 按用户给的理论表查漏补缺，新增 av_memory_language_v2：
  Reynolds-Heeger 乘性注意场（v1 的相加式会让弱刺激候选被目标项抬起来）+ Selective Tuning 方位抑制环
  （两人并排时不再来回拉扯焦点；无方位则不抑制）。交流意愿层抽到 attention_common.hpp 由 v1/v2 共享。
  v1/v2 跑同一套行为用例全过，但现场优劣未比较，默认仍是 v1。方位走 ABI 预留的 azimuth_deg 扩展位，未改布局。
- 识别模型：声纹 CampPlus -> ERes2NetV2（192 维，14.1ms -> 53.9ms/3 秒，句末计算）。
  离线分数分布实测两模型尺度接近（冒充者上限都 0.70、同音色下限都 0.85），原阈值 0.48/0.6 可沿用——
  之前"阈值不可跨模型平移"的担心经测量并不成立。合成语音不等于真人，现场 ROC 仍未做。
  人脸新增可切换后端：sface（默认，Apache-2.0）/ arcface（buffalo_l w600k_r50，更准）。
  默认没改 arcface，因为其权重仅限非商业研究用途而本系统用途未定；用户确认后一个开关即可切换。
  身份识别加 300ms 节流，按人脸位置缓存，身份本就不逐帧变化。
- 旧声纹与人脸档案已按用户指示清空，备份在 ~/Golands/.identity-backup-*；必须真人重新登记。
- 验证：Attention 151、Brain 62、Vision 16、C++ 4 个测试目标、DOM、九场景离线端到端全过；未开硬件。

## 按理论链补齐统一注意力竞争（2026-09-16，最新）
用户给出完整理论链并要求补齐。逐条核对后，真正缺的是三样，已补：
- 跨模态竞争：v2 归一化池改为 本模态 + 0.35×另一模态。此前视听各自归一化，两个模态从不真正争夺有限注意资源。
- Memory → Attention 内源性候选：memory_candidates.py。保守版（用户选择）：只提 Brain 明确未完成的事
  （过期未被回答的问题），同一件只提一次、每 5 分钟最多一次、相关的人必须在场、必须安静 3 秒、30 分钟后不再提；
  每次拒绝都记原因。只放候选进竞争，说不说仍由 Brain 决定。ATTENTION_MEMORY_EVENTS=0 可关。
- Brain → Attention 目标通道：attention_memory.goals（awaiting_answer/holding_floor/deferred_turn/greet_owner）
  转成该人的 importance，只偏置不放行、不能创造未观测到的人（有测试固定）。
另补完两个半成品：
- workspace 升级为认知层竞争：soft-WTA + 有限容量 4 + urgency≥0.7 抢占，非赢家标 suppressed_over_capacity。
  非感知候选（记忆、惊跳）在这里竞争，因为它们没有模态。
- 人脸商用路线：默认 sface + face_quality 质量门槛 + 5 帧模板均值。质量不过就返回 unknown 并给原因，不猜。
明确不做：层级 Selective Tuning。候选空间是扁平的 person/track/object，没有 part-whole 层级，硬套是空架子。
这是判断，不是遗漏；等以后真有层级结构再说。
验证：Attention 164、Brain 62、Vision 23、C++ 4 个目标、DOM、九场景离线端到端、runtime smoke 全过；未开硬件。

## 新层为什么原先兼容不掉旧层（2026-09-16，最新）
用户问「视听融合是否已被新注意力机制完全兼容」。核查结论：证据层完全进来了，决策层原先没有。
实测出一个真实的洞：走过来看着机器人说话（最基本的情况）交流意愿渐近值只有 0.7013，
永远到不了 0.72，全靠 InteractionFusion 的离散快通道撑着。两个结构性原因：
1) 连续层只有 tonic 通道 τ=1.2s，物理上产生不了一两百毫秒的响应，调权重无用；
2) e_av=max(注视×朝向, 唇动×发声) 把「只是看」0.81 与「看着并说话」1.0 压得太近。
补法（不是调参）：
- 新增视听 phasic 通道：extra 键 av_sync_onset，主机在跨模态绑定成立且 gaze≥.65/body≥.6 的那一帧发一次，
  插件施加 1.3×注视×朝向×唇动×发声 的脉冲。只是看 → 无起始 → 无脉冲。
- tonic 改为相乘：e_av = 注视×朝向 × (1 + 0.95×唇动×发声)。说话只在面向我们成立的程度上才算证据。
结果：走过来说话 50ms 达 ENGAGED 稳态 0.796（快于离散路径的 120ms）；只是盯着看 0.666 不变；
旁人说话没看机器人从 0.708（几乎贴着阈值）降到 IDLE。
现状：离散路径 2/3/4/7 在决策上已冗余，但本轮未删除——删除是行为变更，应在一次现场数据之后再做。
反射/惊跳那条保留，因为它必须在插件加载失败时仍然有效，这是安全属性不是理论缺口。
离散路径仍承担「绑定」（source_track_id / 人脸 / adopt 视觉目标），finalize_gate 依赖它，那不是重复裁决。
新增回归用例：cpp_attention_plugin_test 里 walking_up_and_speaking / speech_while_turned_away（v1、v2 各跑一遍），
verify_attention_algorithm.py 增加同名场景。Attention 164、Brain 62、Vision 23、C++ 4 目标、DOM 全过。

## TTS 重复回话的定位与修复（2026-09-16，最新，回归由我引入）
现场现象：同一句回复每 5–7 秒重复，末段退化到每 0.5 秒一次。
定位：hri-memory events.jsonl 的 brain_trace 显示 attention_trigger_started 紧跟每个 turn_finished，
自我循环（246298 触发 → 250105 说完 → 250209 再触发 → 255220 说完 → 255238 再触发）。
根因：把 GazeInvitation 换成 AttentionTrigger 时，去重键从（人, 对话片段 episode_id）
改成了（状态, 人, transition_id）。transition_id 在机器人自己说话时必然变化——
注意力进 RESPONDING 再回 INVITED 就是新 id、新键、重新触发。粘性键被换成了易变键。
修复：不再用任何来自注意力状态的键。重新武装要求「自上次开口以来有新的真人最终转写」，
外加 8 秒最小间隔兜底。关键细节：不能用 last_human_ms（来自原始 VAD），
机器人自己播放时 VAD 也会 active，回声会把它重新武装——改用 heard 列表里最新的非回声转写时刻
（_last_heard_human_ms），回声在进 heard 之前就被拦掉了。
验证：用现场记录的 13 个触发时刻回放，修复后只触发 1 次，其余全部被 nothing_new_since_our_last_turn 拦下。
新增回归用例 test_the_robot_speaking_cannot_re_arm_it（显式覆盖 transition_id 变化不得重新武装）。
Brain 64、Attention 164、C++ 4 目标通过。

## 手势与表情接入注意力（2026-09-16，最新）
用户提出两个场景：主人离开后回来要结合语境+表情手势判断并主动搭话；主人未离开但长时间沉默，
要根据情绪主动询问。核查发现表情和手势此前完全没有进入注意力计算（competition_adapter /
interaction_fusion / attention_common.hpp 零引用），只到千问那里组织措辞。已按「不同问题进不同层」接入：
- 手势进交流意愿层的 phasic 通道。招手/叫人是视觉版的叫名字：有意图、离散、作为稳态无意义。
  extra 键 gesture_invite/gesture_reject，主机在手势首次识别的那一帧发一次（同一手势不重复计分）。
  招手 +2.2×分数×朝向，摆手拒绝 −2.4×分数（权重更高，认错「别烦我」代价更大）。
  实测招手维持约 1.6 秒 ENGAGED，摆手压到 0。脉冲只施加一次但 reason 位保留 3 秒，否则日志解释不了信念跳变。
- 表情不进交流意愿层。理由：交流意愿回答「他是不是在跟我说话」，表情在这个问题上是弱证据，
  掺进去等于见谁笑就搭话。改为工作区内源性候选 affect_candidates.py，回答「谁看起来需要我开口」：
  20 秒窗口 ≥12 个 valid 样本、≥60% 负面、已安静 ≥20 秒、识别置信度 ≥0.5、每人每 10 分钟最多一次；
  负载给出样本数/占比/均值/安静时长，措辞是「表情持续偏低」不是断言心情。ATTENTION_AFFECT_EVENTS=0 可关。
- 人离开 ≥3 秒后再出现触发去习惯化（change_id="return:<人>:<时刻>"）：回来本身是变化，应重新变得显著。
- 补上了之前断掉的一段链路：工作区里 admitted 的非感知候选经 state / brain_input 的 workspace_events
  交给 Brain，Brain 用与无声邀请相同的门槛（busy、8 秒间隔、自上次开口以来有新真人转写）决定是否开口。
  被容量压制的候选不会交出去。prompt_context 按候选类型给不同提示词，明确要求不断言对方心情。
限制：表情分类器噪声大，阈值（12 样本/60%/20 秒）是工程起点未在真人标定；
主人「离开又回来」的主动问候仍是 Brain 的 _owner_presence 规则，尚未并入交流意愿层。
验证：Attention 170、Brain 68、Vision 23、C++ 4 目标、DOM、离线端到端、runtime smoke 全过。

## 沉默+情绪主动询问，以及主人问候合并的中止（2026-09-16，最新）
可测试的部分（已端到端验证，未开硬件）：
- affect_candidates.py 有一个必须修的缺陷已修：MIN_SAMPLES=12 在视觉 20fps 下只有 0.6 秒，
  那是「皱了下眉」不是「持续偏低」。加了 MIN_SPAN_MS=8000，实测真实速率下首次触发从 0.6s 变成 8.0s。
- 链路已通：affect/memory 候选 -> 工作区 admitted -> state 与 brain_input 的 workspace_events ->
  Brain AttentionTrigger（busy / 8 秒间隔 / 自上次开口以来有新真人转写）-> prompt_context 按类型给提示词。
  affect 的提示词明确要求不断言对方心情、不提「我看你表情」。Brain 端有 runtime 级用例覆盖。
发现并修复的真 bug：交流意愿积分器用精确解，dt 很大时会一步跳到稳态，
等于给「没观察到的那段时间」也算了证据（测试里跳过 19 秒后 50ms 就 INVITED）。
已加 kMaxRisingDt=1.0：遗忘按真实 dt，累积不得超过实际观察到的时间。卡顿或人离开后回来都不会再瞬间越阈值。
修正后实测：主人回来（本次聊过+跨会话熟悉）1.3 秒 INVITED，（只有跨会话熟悉）2.5 秒，陌生人永不。

主人「离开又回来」的主动问候合并进注意力：**尝试过，中止，已回滚开口部分**。
可行性已验证：INVITED 确实覆盖了这个场景（1.3–2.5 秒）。中止原因：
_owner_presence 承载 7 条用户明确要求过的保证（必须锁定、连续安静5秒、视觉短暂丢失保留等待不开口、
身份置信度波动不重复、锁定丢失取消、注意力切换重置等待），合并后这些保证的归属方变成注意力层，
需要删改 7 个 Brain 测试并搬走 state["initiative"] 面板状态——是结构调整不是补丁。
在现场测试之前做，等于让现场验证一份几分钟前写的代码，且安全网被拆掉一半。
保留下来的部分：AttentionTrigger 新增 arrival_needs_a_quiet_room（没在本次会话说过话的人，
开口前要求房间安静 5 秒）——这条保证本来只有 _owner_presence 有，现在对任何人都成立；
以及 spoke_this_session 与对应的「刚出现/刚回来」提示词分支。
必须现在修的真缺陷（本轮引入）：加了跨会话熟悉度之后，回来的主人会同时走 _owner_presence 和 INVITED
两条路，同一次到场会开口两次。已修：_owner_presence 开口时同步写 attention_trigger.last_fired_ms，
两条路互斥。新增回归用例 test_the_owner_rule_and_the_invitation_cannot_both_open_the_same_arrival。
验证：Attention 171、Brain 72、Vision 23、C++ 4 目标、DOM、离线端到端全过。

## 注意力算法当前默认与切换方式（2026-09-16，最新）
默认 av_memory_language_v1（未标定，v1/v2 现场优劣未比较，所以没把 v2 设默认）。三种切换方式：
1. 8092「注意力输入源与算法」下拉选 v2 → 应用。写入 .run/attention-config.json，重启保留。
2. 启动时环境变量 ATTENTION_ALGORITHM=av_memory_language_v2。
3. POST /api/attention-config {"algorithm":"av_memory_language_v2"}。
顺带修了两个会坑人的地方：
- 运行中切换算法时，被切入的插件实例还带着上次运行留下的习惯化、返回抑制和交流意愿。
  拿当前帧去和一个已经不存在的情形比较是错的。select() 现在会重建实例，切进去拿到的是算法本身，
  不是它对上一段时间的记忆。新增用例 test_switching_gives_a_fresh_algorithm_not_its_earlier_state。
- 之前持久化配置会静默覆盖 ATTENTION_ALGORITHM，设了环境变量像没生效。
  现在显式环境变量优先于面板保存的选择（显式压过粘性），并且 /api/attention-config 里
  algorithm.chosen_by 会说明这次是 default / environment / saved_console_choice 哪一种。
验证：Attention 173、Brain 72、C++ 4 目标、DOM 通过；三种切换方式各实测一次。

## 现场两个场景的定位与修复（2026-09-16，最新，均为我的 bug）
场景A：镜头外叫名字触发了，但过一会儿再说话就不回了。
  复现：交流意愿只有 0.678（阈值 0.72），而且融合层根本没有 ENGAGED 分支——只有 EXPECTED_ANSWER 和 INVITED。
  根因两条：
  1) 记忆通道没有区分「说过话」和「在对机器人说话」。recent_participants 两者都算作参与者，
     所以刚跟机器人对话完的人和旁边闲聊的人权重一样。
  2) interaction_fusion 没有 ENGAGED 分支，即使意愿够了也不放行。
  修复：Brain 新增 _recent_addressees()，attention_memory.recent_participants 每项带 last_addressed_ms；
  attention_sources 衰减成 memory_open_exchange=exp(-age/15000)（60 秒外丢弃），
  经 extra 键 memory_open_exchange 进插件，权重 0.40，并纳入语言 phasic 的合取项。
  融合层新增 ENGAGED 分支：可靠声纹（≥.6）的同一人再次开口即放行，不要求可见。
  实测：3 秒前对机器人说过话 → 1350ms 放行；10 秒 → 1950ms；30 秒以上不放行；
  从没对机器人说过话的旁人永不放行。约 15 秒跟随窗口，与车载语音的 follow-up 行为一致。
场景B：旁人在说话时有人转头看机器人，不回应。
  复现：交流意愿层两种情况都正确给出 INVITED 0.773，是我在融合层加的 `not clean`
  （「没有任何干净语音」）把它扔掉了。应该是「这个人自己没在说话」，而不是「没人在说话」——
  交流意愿层本来就已经要求该人 voice_activity<0.3 才报 INVITED。
  修复：改为只检查归属该人的干净声轨。原测试断言的正是错误行为，已改写为断言真正该保证的那条。
顺带修掉一个潜在 bug：手势 reason 位用 `stamp - invited_ms < 3000` 判断，默认值 0 在小时间戳下落在窗口内，
纯音频候选也会报 gesture_invite/gesture_reject。已加 >0 守卫。
验证：Attention 176、Brain 72、Vision 23、C++ 4 目标、DOM、离线端到端全过。

## 接上 Sipeed 后「对着镜头说话没反应」的定位（2026-09-16，最新，我的 bug）
现象：切到 v2 后对着镜头说话不回，画面检测到人、ASR 也成功。
先排除 v2：基本场景 v1 50ms/0.796、v2 50ms/0.756，两者都正常。与 v2 无关。
真因：跨模态绑定条件写反了。原条件是
  (声源无方位 且 只有一个人) 或 (两边都有方位且相差<30°)
「只有一个人」这个兜底**只在声源没有方位时生效**。用 Mac 内置麦时声源恒为无方位，一直走兜底；
插上 Sipeed 后声源有了方位（实测值在 -90/-60/120 之间乱跳，阵列未标定），兜底失效，
只要视觉没方位或两边对不上，绑定就永不成立 —— 唇音同步、跨模态增益、起始脉冲全部消失。
修复：一张动的嘴 + 一个声音就是绑定，方位只用来区分人群，不是前提；
未标定阵列的方位分歧更可能是测角误差而不是一个看不见的第二说话人。
单人单声轨时即使方位差 70° 也绑定，并把分歧记进 binding_notes 供排查；
两人及以上时方位仍然按 ≥30° 拒绝（有用例固定）。interaction_fusion 的离散路径同一条件同样修正。
为什么 v1 那次还能回：不是 v1 的功劳，是当时已登记声纹。
finalize_gate 有一条 confirmed_voice_at_engaged_target：可靠声纹(≥.6) + 持续 VISUAL_FOCUS 即可放行最终句，
不需要跨模态绑定。实测：相似度 .9 放行，陌生人(.2) 拒绝。
所以清空档案之后，陌生人这条路也没了，绑定又是断的，于是完全不回。
验证：Attention 179、Brain 72、Vision 23、C++ 4 目标、DOM、离线端到端全过。
