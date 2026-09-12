#pragma once

#include "voice_detection/types.hpp"

#include <optional>
#include <vector>

namespace voice_detection
{

std::vector<float> mixdown(
  const AudioFrame & frame,
  const std::optional<std::vector<int>> & channel_indices = std::nullopt);

std::vector<float> delayAndSum(
  const AudioFrame & frame,
  const MicrophoneProfile & profile,
  float azimuth_deg,
  float speed_of_sound_m_s = 343.0F);

}  // namespace voice_detection

