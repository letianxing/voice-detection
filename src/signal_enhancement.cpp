#include "voice_detection/signal_enhancement.hpp"

#include "voice_detection/math_utils.hpp"

#include <cmath>

namespace voice_detection
{

AdaptiveNoiseSuppressor::AdaptiveNoiseSuppressor(float noise_alpha, float max_reduction)
: noise_alpha_(noise_alpha), max_reduction_(max_reduction)
{
}

LinearEchoCanceller::LinearEchoCanceller(float smoothing, float max_gain)
: smoothing_(clamp(smoothing, 0.0F, 0.999F)), max_gain_(std::max(0.0F, max_gain))
{
}

std::vector<float> LinearEchoCanceller::process(
  const std::vector<float> & microphone,
  const std::optional<std::vector<float>> & playback_reference)
{
  if (!playback_reference.has_value()) {
    return microphone;
  }
  const std::size_t count = std::min(microphone.size(), playback_reference->size());
  if (count < 32) {
    return microphone;
  }
  double dot = 0.0;
  double reference_energy = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    dot += static_cast<double>(microphone[i]) * static_cast<double>((*playback_reference)[i]);
    reference_energy += static_cast<double>((*playback_reference)[i]) *
      static_cast<double>((*playback_reference)[i]);
  }
  if (reference_energy < 1e-8) {
    return microphone;
  }
  const float estimate = clamp(
    static_cast<float>(dot / reference_energy), -max_gain_, max_gain_);
  echo_gain_ = (smoothing_ * echo_gain_) + ((1.0F - smoothing_) * estimate);
  std::vector<float> result = microphone;
  for (std::size_t i = 0; i < count; ++i) {
    result[i] = clamp(result[i] - (echo_gain_ * (*playback_reference)[i]), -1.0F, 1.0F);
  }
  return result;
}

std::vector<float> AdaptiveNoiseSuppressor::process(
  const std::vector<float> & mono,
  bool speech_active_hint)
{
  if (mono.empty()) {
    return {};
  }
  double energy = 0.0;
  for (const float sample : mono) {
    energy += static_cast<double>(sample) * static_cast<double>(sample);
  }
  const float rms = static_cast<float>(std::sqrt((energy / static_cast<double>(mono.size())) + 1e-12));
  if (!speech_active_hint) {
    noise_rms_ = (noise_alpha_ * noise_rms_) + ((1.0F - noise_alpha_) * rms);
  }

  const float snr = rms / std::max(noise_rms_, 1e-6F);
  const float gain = clamp(0.55F + (0.45F * ((snr - 1.0F) / 6.0F)), max_reduction_, 1.0F);
  std::vector<float> out = mono;
  for (float & sample : out) {
    sample *= gain;
  }
  return out;
}

std::vector<float> removeDcPerChannel(const std::vector<float> & samples, int channels)
{
  if (channels <= 0 || samples.empty()) {
    return samples;
  }
  std::vector<float> out = samples;
  const std::size_t frames = samples.size() / static_cast<std::size_t>(channels);
  if (frames == 0) {
    return out;
  }
  for (int channel = 0; channel < channels; ++channel) {
    double sum = 0.0;
    for (std::size_t i = 0; i < frames; ++i) {
      sum += samples[(i * static_cast<std::size_t>(channels)) + static_cast<std::size_t>(channel)];
    }
    const float mean = static_cast<float>(sum / static_cast<double>(frames));
    for (std::size_t i = 0; i < frames; ++i) {
      out[(i * static_cast<std::size_t>(channels)) + static_cast<std::size_t>(channel)] -= mean;
    }
  }
  return out;
}

}  // namespace voice_detection
