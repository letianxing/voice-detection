# Voice 声学反射、音乐与正式大脑接口

## 已接入的处理链

`LiveMonitor` 把录音帧送入独立、有界队列的 `SceneRuntime`；反射和方向在快速工作线程计算，AST/曲风推理在独立进程运行，不在音频回调里加载模型。原来的 ASR、说话人登记和注意力打断链路保留。

- 声学：RMS dBFS、可选校准 SPL、阵列方向、方向可信度、输入年龄和丢帧计数。
- 快反射：响声突增触发 `startle`，后方持续声音触发 `orient`，有启动标定、不应期、自回声和已检测音乐抑制。单麦可惊跳，但不伪造方向，不触发有方向的后方转向。
- 音乐存在性：[MIT AST AudioSet](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)。
- 曲风：[dima806/music_genres_classification](https://huggingface.co/dima806/music_genres_classification)，十类为 blues/classical/country/disco/hiphop/jazz/metal/pop/reggae/rock。默认概率分数阈值0.90、第一二名差0.15；不足则 unknown。分数不是已校准正确率，模型也不能保证识别训练标签以外的曲风。
- BPM：librosa 起音包络和 beat tracking，并要求周期性/规律度通过；静音和无稳定节拍返回无效。音高不是 BPM，不能据 pitch 计算 BPM。
- 节拍：在已确认有音乐且 BPM 有效时，结合估计节拍相位和最小间隔，输出落在节拍附近的观测起音脉冲；无输入时不会凭空持续发节拍。
- 语义意图：对通过注意力门控的精确短指令识别 follow/stop/hello；不是任意自然语言意图分类器。

首次音乐判断需要至少4秒音频，滚动窗口最长10秒，每2秒提交一次分析。安静持续500ms会清除音乐状态；过期结果不会覆盖新静音。曲风、BPM、声源位置都需真实场地验收。

## 安装与启动

当前机器依赖和两个模型已安装。换机器时：

```bash
cd ~/Golands/voice-detection
../robot-attention-perception/.venv-mac/bin/python -m pip install -e '.[live,scene]'
../robot-attention-perception/.venv-mac/bin/python scripts/setup_audio_perception.py
```

下载的是固定 revision 的 safetensors 文件，记录 SHA256 和来源；运行时仅从本地加载，不下载或执行远程模型代码。
模型分别保存在 `weights/ast-audioset`、`weights/music-genre`，合计约700MB量级。

统一启动用 `biomimetic-brain-test/scripts/start_all.sh`；分开启动用 `scripts/start_component.sh`，两者共用模型、端口与会话配置，并自动传入 Voice 的 ROS bridge 地址。

ROS镜像现在通过 Docker build additional_contexts 读取已有的
`../pacific-rim/module/service/robo_brain_service/ros2/src/audio_msgs`，没有在 Voice 或 Attention 源码里再复制一份消息定义。更新后预构建一次：

```bash
cd ~/Golands/robot-attention-perception
docker compose -f docker-compose.mac.yml build
```

单独运行Voice时（ROS已启动）：

```bash
cd ~/Golands/voice-detection
VOICE_ROSBRIDGE_URL=ws://127.0.0.1:9090 \
../robot-attention-perception/.venv-mac/bin/python scripts/local_voice_dashboard.py
```

8090选择设备和profile并点击Start；Mac单麦没有方向，MicArray选 `sipeed_6_plus_1_usb_array`，并验证实际通道/安装方向。

## 输出与协议

配置在 `config/brain_topics.json`：

| Topic | 类型 | 含义 |
|---|---|---|
| `/voice_msg` | std_msgs/msg/String（JSON） | 门控后的ASR、角色、稳定数字ID、原始speaker_label、浮点声纹、角度有效性 |
| `/sound_direction` | std_msgs/msg/Float32 | 仅在方向有效且新鲜时发布 |
| `/startle_trigger` | std_msgs/msg/Bool | 惊跳事件脉冲 |
| `/rear_turn_trigger` | std_msgs/msg/Bool | 带有效方向的转向事件脉冲 |
| `/audio/music_state` | std_msgs/msg/Bool | 音乐状态；失效/停止时清除 |
| `/music_genre` | std_msgs/msg/String | 曲风或unknown |
| `/music_bpm` | std_msgs/msg/Float32 | 有效估计；清除时为0 |
| `/audio/music_beat` | std_msgs/msg/Bool | 观测节拍事件 |
| `/audio_intent_result` | audio_msgs/msg/AudioIntent | 精确短指令intent/prob |
| `/voice/perception_state` | std_msgs/msg/String（JSON） | 完整状态、有效性、模型错误、原始最新转写 |

默认ASR使用正式大脑已经订阅的 `/voice_msg` JSON路径，保留未知角度、VAD和原始ID等信息。它与 `/voice_msg_speaker` 是替代入口，不同时重复发送一条ASR。若确需 typed AudioMsg，配置 `voice_topic=/voice_msg_speaker`、`voice_type=audio_msgs/msg/AudioMsg`；因该类型无角度有效性字段，无方向的语句将不走此入口，仍可从原始状态读取。

默认发布给正式大脑的ASR/语义意图需8092中同一句转写的 `listen=true + addressed_to_robot=true`；旁人对话仍在Voice原始转写和原有记忆链路中。反射/音乐不受对话资格门限制。

坐标采用已核实的正式动作端约定：正式大脑 `direction-90` 转为头部yaw，因此发送端以 `legacy_angle=normalize(robot_yaw+90)` 发布，正前方为90。Voice内部 `direction_deg` 正前方0、正向沿阵列坐标+y（默认逆时针）。用 `VOICE_DOA_SIGN`（默认1）和 `VOICE_DOA_OFFSET_DEG`（默认0）校准安装方向。不要直接把未知值填0。

SPL与dBFS不同。默认只给出dBFS，`spl_db=null`；有实际声压校准后才设置 `VOICE_SPL_CALIBRATION_DB`。反射使用相对dBFS变化，不假装是校准声压计。

ID映射保存在 `config/brain_speaker_ids.json`，跨重启稳定；原始person1/owner仍以speaker_label保留。若与其他设备联合，需共用身份映射而不是各自随意编号。

## 正式大脑修复

在 pacific-rim 的 robo_brain_service 内完成：

- ROSBridge与CDR路径均保留float32声纹，不再转int16；身份处理与聚合也保持float32。
- 保留原始speaker_label，解析出VAD，保留显式angle_valid=false；未知角度不作为0度参与gaze。
- 方向缓存超过1秒失效；有明确未知方向的新说话人输入，不借用上一说话人的方向。
- 角度说明与动作端90度正前方约定统一。

未手改公共IDL或生成的公共绑定；使用现有标准消息/已有audio_msgs包。手写的服务内CDR适配代码已同步修复。

`biomimetic-brain-test` 的8094新增声学/音乐状态及反射/节拍意图日志，并把事件记入HRI；它仍不驱动实体电机。

## 验证

```bash
cd ~/Golands/voice-detection
../robot-attention-perception/.venv-mac/bin/python -m unittest discover -s tests
../robot-attention-perception/.venv-mac/bin/python scripts/analyze_audio_scene.py /path/to/music.wav
../robot-attention-perception/.venv-mac/bin/python scripts/verify_scene_runtime.py
```

现场：普通讲话不能冒充音乐；播放已知曲目检查genre/BPM有效性；音乐停止后状态清除；拍手检查惊跳不重复刷屏；阵列后方说话检查方向和orient；机器人外放应尽量不触发自身反射。`audio_scene.error`、`dropped_frames`、`processing_age_ms`、`ros.error`均可在8090/api/state检查。

自动和录音回放不能替代实际房间中混响、麦克风几何、音量与真实多人重叠测试。
