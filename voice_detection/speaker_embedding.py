from __future__ import annotations

import json
import uuid
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class SpeakerEmbedder:
    """Optional sherpa-onnx speaker encoder with cosine matching."""

    def __init__(self, model_path: str | Path, num_threads: int = 2):
        self.model_path = Path(model_path).expanduser()
        self.extractor = None
        self.error = ""
        try:
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model_path),
                num_threads=int(num_threads),
                provider="cpu",
            )
            self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        except Exception as exc:
            self.error = str(exc)

    @property
    def ready(self) -> bool:
        return self.extractor is not None

    @property
    def dim(self) -> int:
        return int(self.extractor.dim) if self.extractor is not None else 0

    def embed(self, samples: np.ndarray, sample_rate_hz: int) -> list[float] | None:
        if self.extractor is None:
            return None
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size < max(1600, int(sample_rate_hz * 0.5)):
            return None
        if sample_rate_hz != 16000:
            audio = resample(audio, sample_rate_hz, 16000)
        stream = self.extractor.create_stream()
        stream.accept_waveform(16000, audio)
        if not self.extractor.is_ready(stream):
            return None
        vector = np.asarray(self.extractor.compute(stream), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return (vector / norm).tolist() if norm > 1e-6 else None

    def enrollment_embedding(self,samples,sample_rate_hz):
        audio=np.asarray(samples).reshape(-1)
        if len(audio)<sample_rate_hz*2:
            raise ValueError("请完整说至少两秒的注册话术")
        vectors=[self.embed(part,sample_rate_hz) for part in np.array_split(audio,2)]
        if any(v is None for v in vectors) or float(np.dot(vectors[0],vectors[1]))<.6:
            raise ValueError("两段声纹不一致，请一人清晰说完注册话术")
        mean=np.mean(np.asarray(vectors),axis=0)
        mean/=max(float(np.linalg.norm(mean)),1e-6)
        return mean.tolist()


class SpeakerProfileStore:
    """Enrolled voices, and the decision about whether this one is any of them.

    Raw cosine is not a transferable measure. Measured on this machine over 70
    recordings from one field session, every ERes2NetV2 embedding carries a
    large common component — mean cosine 0.62 against the cohort mean, and mean
    0.38 between clips of *different* people — so an absolute 0.48 bar rejects
    the owner as soon as the distance, gain or room changes, which is exactly
    what it did. The transferable question is how far above the imposter cohort
    this score stands (AS-norm: Matejka et al., "Analysis of Score
    Normalization in Multilingual Speaker Recognition", Interspeech 2017).

    So there are two ways to be named: an absolute cosine high enough to speak
    for itself, or a score that stands clearly above the other voices this
    machine knows. Both still need a margin over the runner-up, and neither can
    fire below RAW_FLOOR — a normalised score built on noise is still noise.
    """

    MATCH_THRESHOLD = .48    # absolute cosine that names somebody on its own
    RAW_FLOOR = .34          # nobody is named below this, whatever the z score
    Z_FLOOR = 2.2            # standard deviations above the imposter cohort
    MARGIN_FLOOR = .06       # over the runner-up, so two enrolled people never swap
    COHORT_TOP_K = 10        # adaptive cohort size
    COHORT_MIN = 6           # fewer known voices than this and z means nothing
    SIGMA_FLOOR = .03        # keeps a tight cohort from manufacturing huge z

    def __init__(self, path: str | Path):
        self.lock=threading.RLock()
        self.path = Path(path).expanduser()
        self.profiles: dict[str, dict[str, Any]] = {}
        self.thresholds = {"threshold": self.MATCH_THRESHOLD, "raw_floor": self.RAW_FLOOR,
                           "z_floor": self.Z_FLOOR, "margin_floor": self.MARGIN_FLOOR}
        self.last_decision: dict[str, Any] = {}
        # 上一次「人脸教声纹」的结果：补进去了，还是为什么没补。
        # Without it, a pool that never grows looks exactly like one that has
        # nothing to learn from.
        self.last_pool_reason: str = '还没有过人脸确认的样本'
        # Face-verified samples waiting for a second opinion. In memory only:
        # a sample nothing has corroborated yet does not belong on disk.
        self.pending: dict[str, list] = {}
        # Identities this copy removed. Without it, merging somebody else's
        # writes back in on save resurrects everything the pruner just dropped.
        self.forgotten: set[str] = set()
        self.load()

    def tune(self, **values) -> dict[str, float]:
        """Live threshold changes from the panel; unknown names are ignored."""
        with self.lock:
            for name, value in values.items():
                if name in self.thresholds and value is not None:
                    self.thresholds[name] = float(value)
            return dict(self.thresholds)

    def load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.profiles = payload if isinstance(payload, dict) else {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            self.profiles = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self._merge_from_disk()
            fd,name=tempfile.mkstemp(prefix='.voices-',dir=self.path.parent)
            try:
                with os.fdopen(fd,'w') as output:json.dump(self.profiles,output,ensure_ascii=False,indent=2)
                os.replace(name,self.path)
            finally:
                if os.path.exists(name):os.unlink(name)

    # How many auto-created clusters may exist at once. Unbounded, they became
    # the problem they were meant to solve: twenty singleton clusters from one
    # session, each one another rival in the margin test, so every miss made
    # the next match harder.
    MAX_STRANGERS = 12

    # A voice this close to the robot's own is the robot, and must never become
    # a person. In the field the cluster stranger_a4b2ab9a794d was built out of
    # the robot's playback over two sessions, then matched its own echo at
    # 0.494 and put it in memory as somebody's sentence. Measured against the
    # enrolled synthesiser voice, that cluster scored 0.458 to 0.514.
    ROBOT_VOICE_FLOOR = .4

    def sounds_like_the_robot(self, embedding):
        return any(row["role"] == "robot" and row["score"] >= self.ROBOT_VOICE_FLOOR
                   for row in self.rank(embedding))

    def learn_own_voice(self, embedding):
        """Add an utterance we know was ours to the robot's own pool.

        Enrolment is synthesised audio, so it scores 0.78-0.84 against other
        synthesised lines but only about 0.5 against the same voice after it
        has been through the speaker, the room and echo cancellation. The
        recordings that the text match identified beyond doubt are exactly the
        samples that close that gap, and they cost nothing to collect.
        """
        vector=self._unit(embedding)
        with self.lock:
            profile=self.profiles.get('robot')
            if vector is None or not profile or profile.get('role')!='robot':
                return False
            anchor=self._unit(profile.get('embedding') or [])
            if anchor is None or anchor.shape!=vector.shape or float(np.dot(vector,anchor))<.3:
                return False
            try:
                self._enroll('robot',vector.tolist(),'robot',profile.get('enrollment_text',''),anchor_floor=.3)
            except ValueError:
                return False
            return True

    def _merge_from_disk(self):
        """Keep identities another process added while this one was running.

        The whole file is rewritten on every save, so the last writer used to
        win the whole store: enrolling the robot's own voice from a script
        while the service was running put `robot` on disk, and the service's
        next save — which happened to be a new stranger cluster — deleted it
        again and brought back the cluster the script had just removed. An
        identity this copy has never heard of is somebody else's write, and it
        stays.
        """
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        for identity, profile in payload.items():
            if identity in self.forgotten:
                continue      # this copy deleted it on purpose
            if identity not in self.profiles and isinstance(profile, dict):
                self.profiles[identity] = profile

    def remember_stranger(self, embedding):
        speaker_id, role, similarity = self.match(embedding)
        if speaker_id != "unknown":
            return speaker_id, role, similarity
        if self.sounds_like_the_robot(embedding):
            return "robot", "robot", next(row["score"] for row in self.rank(embedding) if row["role"] == "robot")
        ranked=self.rank(embedding)
        if ranked and ranked[0]["score"]>=self.thresholds["threshold"]:
            return "unknown","unknown",ranked[0]["score"]
        speaker_id = "stranger_" + uuid.uuid4().hex[:12]
        self.enroll(speaker_id, embedding, "stranger")
        with self.lock:
            self._prune_strangers()
        return speaker_id, "stranger", 1.0

    def forget(self, identity):
        """Remove an identity for good, so a later merge does not bring it back."""
        with self.lock:
            self.forgotten.add(str(identity))
            existed = self.profiles.pop(str(identity), None) is not None
            if existed:
                self.save()
            return existed

    def _prune_strangers(self):
        strangers=[(key,profile) for key,profile in self.profiles.items() if profile.get("role")=="stranger"]
        if len(strangers)<=self.MAX_STRANGERS:
            return
        strangers.sort(key=lambda item:(int(item[1].get("last_heard_ms") or 0),int(item[1].get("created_ms") or 0)))
        for key,_ in strangers[:len(strangers)-self.MAX_STRANGERS]:
            self.forgotten.add(key)
            self.profiles.pop(key,None)
        self.save()

    def enroll(self, speaker_id: str, embedding: list[float], role: str = "known", enrollment_text: str = "",
               anchor_floor: float = .55) -> None:
        """anchor_floor is lowered only when something outside the voice — a
        confident face on the same utterance — already says who this is. A pool
        enrolled at one distance cannot otherwise learn the same person at
        another, which is exactly how the owner became a stranger in the field."""
        with self.lock:self._enroll(speaker_id,embedding,role,enrollment_text,anchor_floor)

    def _enroll(self,speaker_id,embedding,role,enrollment_text,anchor_floor=.55):
        vector=self._unit(embedding)
        if vector is None:raise ValueError("invalid speaker embedding")
        embedding=vector.tolist()
        previous=self.profiles.get(str(speaker_id),{})
        aliases=list(previous.get("aliases",[]))
        pool=list(previous.get("embeddings") or ([previous["embedding"]] if previous.get("embedding") else []))
        if pool and any(self._unit(v) is None or len(v)!=len(vector) for v in pool):raise ValueError("speaker pool dimension mismatch")
        if pool and float(np.dot(vector,self._unit(pool[0])))<anchor_floor:
            raise ValueError("voice sample too far from original identity anchor")
        if pool and max(float(np.dot(vector,self._unit(v))) for v in pool if self._unit(v) is not None and len(v)==len(vector))<min(.6,anchor_floor):
            # Both guards move together: when a confident face has already said
            # who this is, "too unlike the pool" has to mean a lot more unlike.
            raise ValueError("existing identity voice samples disagree; refusing overwrite")
        if not pool or max(float(np.dot(vector,self._unit(v))) for v in pool if self._unit(v) is not None and len(v)==len(vector))<.985:
            pool=(pool[:1]+pool[-6:]+[embedding]) if len(pool)>=8 else pool+[embedding]
        if role in {"owner", "known"}:
            query=np.asarray(embedding,dtype=np.float32)
            for old_id, old in self.profiles.items():
                vector=np.asarray(old.get("embedding",[]),dtype=np.float32)
                if old.get("role")=="stranger" and vector.shape==query.shape and float(np.dot(query,vector)/(np.linalg.norm(query)*np.linalg.norm(vector)+1e-6))>=.65:
                    old.clear()
                    old.update(canonical_id=str(speaker_id),role="alias")
                    aliases.append(old_id)
        now_ms=int(time.time()*1000)
        self.profiles[str(speaker_id)] = {
            "aliases": aliases,
            "role": str(role or "known"),
            "created_ms": int(previous.get("created_ms") or now_ms),
            "last_heard_ms": now_ms,
            # Nobody can read stranger_c3b41448ec74 off a screen and tell which
            # voice in the room it is. The ordinal is assigned once and kept.
            "display_name": previous.get("display_name") or (
                self._next_stranger_name() if str(role or "")=="stranger" else str(speaker_id)),
            "embedding": pool[0],
            "embeddings": pool,
            "anchor_groups": int(previous.get("anchor_groups") or 1),
            "pool_version": 1,
            "reference_path":previous.get("reference_path"),
            "model": "3dspeaker-eres2netv2-zh-cn",
            "enrollment_text": enrollment_text,
        }
        self.save()

    def _next_stranger_name(self):
        used={str(profile.get("display_name") or "") for profile in self.profiles.values()}
        ordinal=1
        while f"陌生人{ordinal}" in used:
            ordinal+=1
        return f"陌生人{ordinal}"

    def display_name(self,speaker_id):
        """What to put on a screen for this voice."""
        profile=self.profiles.get(str(speaker_id)) or {}
        if profile.get("role")=="alias" and profile.get("canonical_id"):
            return self.display_name(profile["canonical_id"])
        return str(profile.get("display_name") or speaker_id or "unknown")

    def merge_cluster(self,cluster_id,identity):
        """Fold an auto-created cluster into an enrolled person's pool.

        The user's rule: every new voice gets registered, but a voice that
        turns out to belong to somebody already enrolled has to end up in that
        person's pool rather than living on as a rival. The claim comes from
        outside the voiceprint — a confident face on the same utterance — and
        every sample still passes the enrolment guards, so a mistaken face
        cannot rewrite an identity, only fail to extend it.
        """
        with self.lock:
            cluster=self.profiles.get(str(cluster_id))
            target=self.profiles.get(str(identity))
            if not cluster or not target or cluster.get("role")!="stranger":
                return False
            if target.get("role") not in {"owner","known"}:
                return False
            samples=[v for v in (cluster.get("embeddings") or [cluster.get("embedding")]) if v]
            merged=0
            for sample in samples:
                try:
                    self._enroll(identity,sample,target.get("role","known"),
                                 target.get("enrollment_text",""),anchor_floor=self.FACE_VERIFIED_FLOOR)
                    merged+=1
                except ValueError:
                    continue
            aliases=sorted(set(list(self.profiles[str(identity)].get("aliases",[]))+[str(cluster_id)]))
            self.profiles[str(identity)]["aliases"]=aliases
            self.profiles[str(cluster_id)]={"canonical_id":str(identity),"role":"alias",
                                            "merged_samples":merged}
            self.save()
            return True

    @staticmethod
    def _unit(embedding):
        vector=np.asarray(embedding,dtype=np.float32)
        if vector.ndim!=1 or not vector.size or not np.isfinite(vector).all():return None
        norm=float(np.linalg.norm(vector))
        return vector/norm if norm>1e-6 else None

    def rank(self,embedding):
        query=self._unit(embedding)
        if query is None:return []
        ranked=[]
        with self.lock:
            for identity,profile in self.profiles.items():
                if profile.get('role')=='alias':continue
                scores=[]
                for sample in profile.get('embeddings') or [profile.get('embedding',[])]:
                    vector=self._unit(sample)
                    if vector is not None and vector.shape==query.shape:scores.append(float(np.dot(query,vector)))
                if scores:
                    scores.sort(reverse=True)
                    score=scores[0] if len(scores)==1 else .75*scores[0]+.25*scores[1]
                    ranked.append({'speaker_id':identity,'role':profile.get('role','known'),'score':score,'pool_size':len(scores)})
        ranked=sorted(ranked,key=lambda x:x['score'],reverse=True)
        return self._normalize(ranked)

    def _normalize(self,ranked):
        """Adaptive score normalisation:每个候选相对于其余已知声音的位置。

        The cohort is everybody else this machine knows, strangers included —
        auto-created clusters are the closest thing available to an imposter
        set, and using them is what makes the number comparable across
        recordings taken at different distances and gains.
        """
        for row in ranked:
            cohort=sorted((other['score'] for other in ranked if other is not row),reverse=True)[:self.COHORT_TOP_K]
            if len(cohort)<self.COHORT_MIN:
                row['z']=None
                row['cohort']={'size':len(cohort)}
                continue
            mean=float(np.mean(cohort));std=max(float(np.std(cohort)),self.SIGMA_FLOOR)
            row['z']=round((row['score']-mean)/std,2)
            row['cohort']={'size':len(cohort),'mean':round(mean,3),'std':round(std,3)}
        return ranked

    def identify(self, embedding: list[float], threshold: float | None = None) -> dict[str, Any]:
        """The decision with its reasoning attached, for the panel and the logs."""
        self.match(embedding, threshold)
        return dict(self.last_decision)

    def match(self, embedding: list[float], threshold: float | None = None) -> tuple[str,str,float]:
        ranked=self.rank(embedding)
        with self.lock:
            tuned=dict(self.thresholds)
        threshold=float(tuned['threshold'] if threshold is None else threshold)
        if not ranked:
            self.last_decision={'speaker_id':'unknown','reason':'no_enrolled_voice','thresholds':tuned}
            return 'unknown','unknown',0.
        best=ranked[0]
        # The margin exists so two enrolled people are never confused. An
        # auto-created stranger cluster is not a second person — it is usually
        # this same person recorded at another angle, and in the field nine of
        # them piled up and started blocking the owner's own match: every miss
        # created another cluster, which made the next match harder. Strangers
        # therefore do not compete against a named identity.
        if best['role']=='stranger':
            # A named identity within a hair of an unnamed cluster is the better
            # answer: the cluster has no history and is usually this same person
            # from another angle, and calling them by name is what lets the two
            # merge later.
            named=next((row for row in ranked if row['role'] in {'owner','known'}),None)
            if named and named['score']>=threshold and best['score']-named['score']<=.08:
                best=named
        rivals=[row for row in ranked if row is not best and (best['role']=='stranger' or row['role']!='stranger')]
        margin=best['score']-(rivals[0]['score'] if rivals else 0.)
        z=best.get('z')
        absolute=best['score']>=threshold
        normalized=best['score']>=tuned['raw_floor'] and z is not None and z>=tuned['z_floor']
        self.last_decision={'speaker_id':best['speaker_id'],'role':best['role'],
                            'score':round(float(best['score']),3),'z':z,'margin':round(float(margin),3),
                            'pool_size':best.get('pool_size'),'cohort':best.get('cohort'),
                            'thresholds':tuned,
                            'path':'absolute' if absolute else ('cohort_normalized' if normalized else 'none'),
                            'runner_up':rivals[0]['speaker_id'] if rivals else None}
        if not (absolute or normalized):
            self.last_decision.update(speaker_id='unknown',
                                      reason='below_absolute_and_cohort_bars' if z is not None else 'below_absolute_bar_no_cohort')
            return 'unknown','unknown',best['score']
        if margin<tuned['margin_floor']:
            self.last_decision.update(speaker_id='unknown',reason='another_voice_scores_almost_the_same')
            return 'unknown','unknown',best['score']
        self.last_decision['reason']='named'
        return best['speaker_id'],best['role'],best['score']

    # How unlike the enrolled pool a face-confirmed sample may be and still join.
    # Not zero: a confident face on a badly segmented utterance must not be able
    # to put anybody's voice into this pool.
    FACE_VERIFIED_FLOOR = .35

    # 人脸有多确定，才够格拿这句话去教声纹。Same number the attention service
    # uses to decide whether to offer the sample at all; two different bars
    # only meant posting samples that were refused on arrival.
    FACE_LEARNING_FLOOR = .75

    def add_verified_sample(self,identity,embedding,face_identity,face_confidence,quality):
        """Learn this person's voice from an utterance the face already identified.

        The old rule also required the voiceprint to score .7 against the pool it
        was supposed to be extending, so a pool enrolled face-on at half a metre
        could never learn the same person leaning back in a chair — and in the
        field it never did: the owner scored 0.0 off-axis and was filed as a new
        stranger, which then made the next match harder still.

        The face is the independent evidence. The voice only has to not
        contradict it: no other enrolled person may claim this sample, and it
        must be at least vaguely this person. The audio quality bar stays where
        it was — overlapped or echoing speech teaches the pool the room, not the
        person."""
        floor=float(getattr(self,'face_learning_floor',self.FACE_LEARNING_FLOOR))
        for failed,reason in ((identity!=face_identity,'声纹要教的人和人脸认出的人不是同一个'),
                              (face_confidence<floor,f'人脸只有 {face_confidence:.2f}，低于 {floor:.2f}'),
                              (quality.get('overlap',1)>.15,'这句话有重叠说话'),
                              (quality.get('echo',1)>.2,'这句话里混了机器人自己的播放'),
                              (quality.get('duration_ms',0)<2000,'这句话不够 2 秒，做模板太短')):
            if failed:
                self.last_pool_reason=reason
                return False
        with self.lock:
            profile=self.profiles.get(identity)
            if profile is None:
                # 人脸认得出他，声纹库里却没有他——注册时只成功存了人脸就是这种
                # 情况。以前这里直接 return False，于是这个人永远学不到声纹，
                # 离开镜头就再也认不出来；现场 16:42 那次就是这样。开档的条件
                # 和加锚点一样严：两条互相吻合的、人脸确认过的样本。
                self.last_pool_reason='还没有声纹档案，先攒一条，等第二条吻合的就开档'
                return self._second_opinion(identity,embedding,role='known')
            if profile.get('role') not in {'owner','known'}:
                self.last_pool_reason=f'{identity} 在声纹库里的角色是 {profile.get("role")}，不学'
                return False
            if self.sounds_like_the_robot(embedding):
                self.last_pool_reason='这段听起来是机器人自己的播放'
                return False   # the robot's own playback is not a sample of anybody
            ranked=self.rank(embedding)
            named=[row for row in ranked if row['role'] in {'owner','known'}]
            if named and named[0]['speaker_id']!=identity and named[0]['score']>=.6:
                self.last_pool_reason=f'声纹更像 {named[0]["speaker_id"]}，与人脸矛盾，不学'
                return False   # the voiceprint names a different enrolled person; that is a conflict, not a sample
            score=next((row['score'] for row in ranked if row['speaker_id']==identity),0.)
            if score<self.FACE_VERIFIED_FLOOR:
                self.last_pool_reason=f'与已有声纹只有 {score:.2f}，先攒着等第二条吻合的'
                return self._second_opinion(identity,embedding)
            try:
                self.enroll(identity,embedding,profile['role'],profile.get('enrollment_text',''),
                            anchor_floor=self.FACE_VERIFIED_FLOOR)
            except ValueError:
                self.last_pool_reason='与原始锚点差太远，先攒着等第二条吻合的'
                return self._second_opinion(identity,embedding)
            self.last_pool_reason=f'已补入 {identity} 的声纹池（与已有声纹 {score:.2f}）'
            return True

    # Two face-verified samples that agree with each other open a second anchor
    # group. Measured reason: the owner's enrolment is one 2.7 s clip recorded
    # at -51 dBFS, and his live sentences through the other microphone scored
    # 0.083 against it — rank 18 of 21. No threshold can rescue that, and a
    # single confident face is not enough to let a voice into somebody's
    # identity. Two samples from confident faces that also match each other
    # are: they agree about the person and about the channel.
    SECOND_OPINION_AGREEMENT = .5
    SECOND_OPINION_WINDOW_MS = 600000

    def _second_opinion(self,identity,embedding,role=''):
        vector=self._unit(embedding)
        if vector is None:
            return False
        now=int(time.time()*1000)
        waiting=[(when,held) for when,held in self.pending.get(identity,[]) if now-when<=self.SECOND_OPINION_WINDOW_MS]
        for when,held in waiting:
            other=self._unit(held)
            if other is None or other.shape!=vector.shape:
                continue
            if float(np.dot(vector,other))>=self.SECOND_OPINION_AGREEMENT:
                self.pending[identity]=[]
                return self._admit_anchor_group(identity,[held,vector.tolist()],role=role)
        self.pending[identity]=(waiting+[(now,vector.tolist())])[-4:]
        return False

    def _admit_anchor_group(self,identity,vectors,role=''):
        """Add samples the existing anchor cannot reach, as their own anchor.

        With `role` given this also opens the file itself, for somebody the
        camera knows and the voiceprint store has never heard: two agreeing
        face-verified samples are the same evidence that earns a second anchor,
        and without it a person enrolled by face alone stays unrecognisable the
        moment they step out of frame.
        """
        profile=self.profiles.get(identity)
        if not profile and role:
            profile=self.profiles.setdefault(identity,{"role":role,"embeddings":[],
                                                       "aliases":[],"enrollment_text":"",
                                                       "created_by":"face_verified_samples"})
            self.last_pool_reason=f'两条人脸确认过的样本互相吻合，给 {identity} 建了声纹档案'
        if not profile:
            return False
        pool=list(profile.get("embeddings") or [profile.get("embedding")])
        pool=[v for v in pool if v]+list(vectors)
        profile["embeddings"]=pool[:1]+pool[-7:] if len(pool)>8 else pool
        profile["anchor_groups"]=int(profile.get("anchor_groups") or 1)+1
        profile["last_heard_ms"]=int(time.time()*1000)
        self.save()
        return True


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    target_size = max(1, int(round(data.size * target_rate / source_rate)))
    positions = np.linspace(0.0, max(0.0, data.size - 1), target_size)
    return np.interp(positions, np.arange(data.size), data).astype(np.float32)
