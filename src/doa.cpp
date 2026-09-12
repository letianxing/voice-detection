#include "voice_detection/doa.hpp"

#include "voice_detection/math_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace voice_detection
{

float crossCorrelationTau(
  const AudioFrame & frame,
  int left_channel,
  int right_channel,
  int max_shift_samples)
{
  if (frame.frames() == 0 || left_channel < 0 || right_channel < 0 ||
    left_channel >= frame.channels || right_channel >= frame.channels)
  {
    return 0.0F;
  }

  float best_score = -1.0F;
  int best_shift = 0;
  for (int shift = -max_shift_samples; shift <= max_shift_samples; ++shift) {
    double cross = 0.0;
    double left_energy = 0.0;
    double right_energy = 0.0;
    int count = 0;
    for (std::size_t i = 0; i < frame.frames(); ++i) {
      const int j = static_cast<int>(i) - shift;
      if (j < 0 || j >= static_cast<int>(frame.frames())) {
        continue;
      }
      const float left = frame.at(i, left_channel);
      const float right = frame.at(static_cast<std::size_t>(j), right_channel);
      cross += static_cast<double>(left) * static_cast<double>(right);
      left_energy += static_cast<double>(left) * static_cast<double>(left);
      right_energy += static_cast<double>(right) * static_cast<double>(right);
      ++count;
    }
    if (count == 0) {
      continue;
    }
    const float score = static_cast<float>(
      std::abs(cross) / (std::sqrt(left_energy * right_energy) + 1e-9));
    if (score > best_score) {
      best_score = score;
      best_shift = shift;
    }
  }
  return static_cast<float>(best_shift) / static_cast<float>(frame.sample_rate_hz);
}

DoaEstimate estimateAzimuthGcc(
  const AudioFrame & frame,
  const MicrophoneProfile & profile,
  float speed_of_sound_m_s,
  float grid_step_deg)
{
  if (!profile.supports_doa || profile.doa_channel_indices.size() < 2 ||
    profile.mic_positions_m.size() < profile.doa_channel_indices.size())
  {
    return {};
  }

  struct PairDelay
  {
    int left;
    int right;
    float tau;
  };

  std::vector<PairDelay> measured;
  for (std::size_t i = 0; i < profile.doa_channel_indices.size(); ++i) {
    for (std::size_t j = i + 1; j < profile.doa_channel_indices.size(); ++j) {
      const int left = profile.doa_channel_indices[i];
      const int right = profile.doa_channel_indices[j];
      const auto & lp = profile.mic_positions_m[static_cast<std::size_t>(left)];
      const auto & rp = profile.mic_positions_m[static_cast<std::size_t>(right)];
      const float dx = lp[0] - rp[0];
      const float dy = lp[1] - rp[1];
      const float dz = lp[2] - rp[2];
      const float baseline = std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
      const int max_shift = std::max(1, static_cast<int>(std::ceil(baseline / speed_of_sound_m_s * frame.sample_rate_hz)));
      measured.push_back({left, right, crossCorrelationTau(frame, left, right, max_shift)});
    }
  }
  if (measured.empty()) {
    return {};
  }

  float best_azimuth = 0.0F;
  float best_error = std::numeric_limits<float>::max();
  for (float azimuth = -180.0F; azimuth < 180.0F; azimuth += grid_step_deg) {
    const float rad = azimuth * static_cast<float>(M_PI) / 180.0F;
    const std::array<float, 3> direction{std::cos(rad), std::sin(rad), 0.0F};
    float error = 0.0F;
    for (const auto & item : measured) {
      const auto & lp = profile.mic_positions_m[static_cast<std::size_t>(item.left)];
      const auto & rp = profile.mic_positions_m[static_cast<std::size_t>(item.right)];
      const std::array<float, 3> diff{lp[0] - rp[0], lp[1] - rp[1], lp[2] - rp[2]};
      const float expected = -((diff[0] * direction[0]) + (diff[1] * direction[1]) + (diff[2] * direction[2])) /
        speed_of_sound_m_s;
      error += std::abs(item.tau - expected);
    }
    error /= static_cast<float>(measured.size());
    if (error < best_error) {
      best_error = error;
      best_azimuth = azimuth;
    }
  }

  float max_delay = 1e-6F;
  for (const auto & item : measured) {
    const auto & lp = profile.mic_positions_m[static_cast<std::size_t>(item.left)];
    const auto & rp = profile.mic_positions_m[static_cast<std::size_t>(item.right)];
    const float dx = lp[0] - rp[0];
    const float dy = lp[1] - rp[1];
    const float dz = lp[2] - rp[2];
    max_delay = std::max(max_delay, std::sqrt((dx * dx) + (dy * dy) + (dz * dz)) / speed_of_sound_m_s);
  }

  return DoaEstimate{
    normalizeAngleDeg(best_azimuth),
    clamp(1.0F - (best_error / max_delay), 0.0F, 1.0F),
    static_cast<int>(measured.size())};
}

}  // namespace voice_detection

