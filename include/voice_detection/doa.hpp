#pragma once

#include "voice_detection/types.hpp"

namespace voice_detection
{

struct DoaEstimate
{
  std::optional<float> azimuth_deg;
  float confidence{0.0F};
  int pair_count{0};
};

float crossCorrelationTau(
  const AudioFrame & frame,
  int left_channel,
  int right_channel,
  int max_shift_samples);

DoaEstimate estimateAzimuthGcc(
  const AudioFrame & frame,
  const MicrophoneProfile & profile,
  float speed_of_sound_m_s = 343.0F,
  float grid_step_deg = 2.0F);

}  // namespace voice_detection

