#include "voice_detection/beamforming.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace voice_detection
{

std::vector<float> mixdown(
  const AudioFrame & frame,
  const std::optional<std::vector<int>> & channel_indices)
{
  std::vector<float> mono(frame.frames(), 0.0F);
  if (frame.frames() == 0 || frame.channels <= 0) {
    return mono;
  }
  std::vector<int> channels;
  if (channel_indices.has_value() && !channel_indices->empty()) {
    channels = *channel_indices;
  } else {
    channels.resize(static_cast<std::size_t>(frame.channels));
    std::iota(channels.begin(), channels.end(), 0);
  }
  for (std::size_t i = 0; i < frame.frames(); ++i) {
    float sum = 0.0F;
    int count = 0;
    for (const int channel : channels) {
      if (channel >= 0 && channel < frame.channels) {
        sum += frame.at(i, channel);
        ++count;
      }
    }
    mono[i] = count > 0 ? sum / static_cast<float>(count) : 0.0F;
  }
  return mono;
}

std::vector<float> delayAndSum(
  const AudioFrame & frame,
  const MicrophoneProfile & profile,
  float azimuth_deg,
  float speed_of_sound_m_s)
{
  if (profile.doa_channel_indices.size() < 2 || profile.mic_positions_m.empty()) {
    return mixdown(frame, profile.doa_channel_indices);
  }

  const float rad = azimuth_deg * static_cast<float>(M_PI) / 180.0F;
  const std::array<float, 3> direction{std::cos(rad), std::sin(rad), 0.0F};
  std::vector<float> delays;
  delays.reserve(profile.doa_channel_indices.size());
  for (const int channel : profile.doa_channel_indices) {
    const auto & pos = profile.mic_positions_m[static_cast<std::size_t>(channel)];
    const float dot = (pos[0] * direction[0]) + (pos[1] * direction[1]) + (pos[2] * direction[2]);
    delays.push_back(-dot / speed_of_sound_m_s);
  }
  const float min_delay = *std::min_element(delays.begin(), delays.end());
  std::vector<float> mono(frame.frames(), 0.0F);
  int used = 0;
  for (std::size_t idx = 0; idx < profile.doa_channel_indices.size(); ++idx) {
    const int channel = profile.doa_channel_indices[idx];
    if (channel < 0 || channel >= frame.channels) {
      continue;
    }
    const int shift = static_cast<int>(std::lround((delays[idx] - min_delay) * frame.sample_rate_hz));
    for (std::size_t i = 0; i < frame.frames(); ++i) {
      const std::size_t src = i + static_cast<std::size_t>(std::max(0, shift));
      if (src < frame.frames()) {
        mono[i] += frame.at(src, channel);
      }
    }
    ++used;
  }
  if (used > 0) {
    for (float & sample : mono) {
      sample /= static_cast<float>(used);
    }
  }
  return mono;
}

}  // namespace voice_detection

