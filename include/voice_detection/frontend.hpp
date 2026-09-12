#pragma once

#include "voice_detection/tracker.hpp"
#include "voice_detection/types.hpp"
#include "voice_detection/signal_enhancement.hpp"
#include "voice_detection/vad.hpp"

#include <optional>
#include <vector>

namespace voice_detection
{

class CocktailFrontend
{
public:
  explicit CocktailFrontend(MicrophoneProfile profile, int vad_calibration_frames = 0);

  FrontendOutput process(
    const AudioFrame & frame,
    const std::optional<std::vector<float>> & playback_reference = std::nullopt);

private:
  MicrophoneProfile profile_;
  EnergyVad pre_vad_;
  EnergyVad post_vad_;
  AdaptiveNoiseSuppressor noise_suppressor_;
  LinearEchoCanceller echo_canceller_;
  SourceTracker tracker_;
};

std::int64_t nowMs();

}  // namespace voice_detection
