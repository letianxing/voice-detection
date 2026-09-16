import sys,time,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_detection.scene_runtime import SceneRuntime
from voice_detection.profiles import load_profiles
from voice_detection.types import AudioFrame

def main():
 scene=SceneRuntime(load_profiles()['mac_builtin'],classifier=True)
 try:
  deadline=time.monotonic()+90
  while time.monotonic()<deadline:
   state=scene.snapshot()
   if state['classifier_ready'] and state['turn_prediction'].get('ready'):break
   if state.get('error'):raise RuntimeError(state['error'])
   time.sleep(.1)
  if not state['classifier_ready'] or not state['turn_prediction'].get('ready'):raise RuntimeError(str(state))
  start=int(time.time()*1000)
  for i in range(180):
   t=(np.arange(960)+i*960)/48000
   x=(.02*np.sin(2*np.pi*200*t)).astype(np.float32)
   scene.submit(AudioFrame(x[:,None],48000,int(time.time()*1000)),x,0.,track_id='offline',vad_active=True,playback_reference=np.zeros_like(x))
   time.sleep(.02)
  time.sleep(.4);state=scene.snapshot()
  print(json.dumps({'classifier_ready':state['classifier_ready'],'error':state['error'],'dropped_frames':state['dropped_frames'],'bio_valid':state['acoustic'].get('bio',{}).get('valid'),'vap':state['turn_prediction']},ensure_ascii=False))
  assert state['acoustic']['bio']['valid'] and state['turn_prediction']['ready'] and not state['error']
 finally:scene.close()
 assert not scene.thread.is_alive() and not scene.process.is_alive() and not scene.turn_predictor.process.is_alive()
 print('all worker processes stopped cleanly')
if __name__=='__main__':main()
