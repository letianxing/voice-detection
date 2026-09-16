#!/usr/bin/env python3
"""Explicit offline overlap transcription; never writes unverified text into memory."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from voice_detection.multi_speaker import MultiSpeakerTranscriber
from voice_detection.speaker_embedding import SpeakerEmbedder,SpeakerProfileStore
from voice_detection.asr import SherpaZipformerAsrAdapter

def main():
 p=argparse.ArgumentParser();p.add_argument('wav');p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 audio,rate=sf.read(a.wav,dtype='float32',always_2d=True)
 store=SpeakerProfileStore(root/'config/speaker_profiles.json')
 embedder=SpeakerEmbedder(root/'weights/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx')
 adapter=SherpaZipformerAsrAdapter(root/'weights/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30')
 try:
  processor=MultiSpeakerTranscriber(store,embedder,adapter,root/'weights/real-tse/pretrained/spk_emb_causal_100')
  result=processor.transcribe(audio.mean(axis=1),rate,0,round(len(audio)*1000/rate))
  Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({'ready':result['ready'],'turns':len(result['turns'])}))
 finally:adapter.close()
if __name__=='__main__':main()
