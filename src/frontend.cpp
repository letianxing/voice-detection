#include "voice_detection/frontend.hpp"

#include "voice_detection/beamforming.hpp"
#include "voice_detection/doa.hpp"
#include "voice_detection/features.hpp"
#include "voice_detection/math_utils.hpp"
#include "voice_detection/signal_enhancement.hpp"

#include <chrono>
#include <cmath>
#include <utility>

namespace voice_detection
{
namespace
{

float estimateClarity(float snr_db)
{
  return clamp(snr_db / 30.0F, 0.0F, 1.0F);
}

float estimateOverlap(const AudioFrame & frame)
{
  if (frame.channels < 2 || frame.frames() == 0) {
    return 0.0F;
  }
  std::vector<float> rms(static_cast<std::size_t>(frame.channels), 0.0F);
  for (int channel = 0; channel < frame.channels; ++channel) {
    double energy = 0.0;
    for (std::size_t i = 0; i < frame.frames(); ++i) {
      const float sample = frame.at(i, channel);
      energy += static_cast<double>(sample) * static_cast<double>(sample);
    }
    rms[static_cast<std::size_t>(channel)] = static_cast<float>(std::sqrt(energy / static_cast<double>(frame.frames())));
  }
  float mean = 0.0F;
  for (const float value : rms) {
    mean += value;
  }
  mean /= static_cast<float>(rms.size());
  float var = 0.0F;
  for (const float value : rms) {
    const float diff = value - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(rms.size());
  const float spread = std::sqrt(var) / std::max(mean, 1e-6F);
  return clamp(1.0F - spread, 0.0F, 1.0F);
}

float estimateSelfEcho(
  const std::vector<float> & mono,
  const std::optional<std::vector<float>> & playback_reference)
{
  if (!playback_reference.has_value()) {
    return 0.0F;
  }
  const std::size_t n = std::min(mono.size(), playback_reference->size());
  if (n < 32) {
    return 0.0F;
  }
  double dot = 0.0;
  double a_energy = 0.0;
  double b_energy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    dot += static_cast<double>(mono[i]) * static_cast<double>((*playback_reference)[i]);
    a_energy += static_cast<double>(mono[i]) * static_cast<double>(mono[i]);
    b_energy += static_cast<double>((*playback_reference)[i]) * static_cast<double>((*playback_reference)[i]);
  }
  return clamp(static_cast<float>(std::abs(dot) / (std::sqrt(a_energy * b_energy) + 1e-9)), 0.0F, 1.0F);
}

}  // namespace

CocktailFrontend::CocktailFrontend(MicrophoneProfile profile, int vad_calibration_frames)
: profile_(std::move(profile)),
  pre_vad_(8.0F, -52.0F, vad_calibration_frames),
  post_vad_(8.0F, -52.0F, vad_calibration_frames)
{
}

FrontendOutput CocktailFrontend::process(
  const AudioFrame & frame,
  const std::optional<std::vector<float>> & playback_reference)
{
  AudioFrame phase_safe_frame = frame;
  phase_safe_frame.samples = removeDcPerChannel(frame.samples, frame.channels);
  const auto raw_mono = mixdown(phase_safe_frame);
  const VoiceActivity pre_activity = pre_vad_.process(raw_mono);
  std::optional<float> azimuth;
  float doa_confidence = 0.0F;
  if (pre_activity.active && profile_.supports_doa && !profile_.doa_channel_indices.empty()) {
    const auto doa = estimateAzimuthGcc(phase_safe_frame, profile_);
    azimuth = doa.azimuth_deg;
    doa_confidence = doa.confidence;
  }

  std::vector<float> spatial_audio = raw_mono;
  if (azimuth.has_value() && doa_confidence >= 0.1F) {
    spatial_audio = delayAndSum(phase_safe_frame, profile_, *azimuth);
  }
  const float self_echo = estimateSelfEcho(spatial_audio, playback_reference);
  const auto echo_cancelled = echo_canceller_.process(spatial_audio, playback_reference);
  std::vector<float> target_audio = noise_suppressor_.process(echo_cancelled, pre_activity.active);
  const VoiceActivity activity = post_vad_.process(target_audio);
  const float clarity = estimateClarity(activity.snr_db);

  FrontendOutput output;
  output.stamp_ms = frame.stamp_ms;
  output.target_audio = target_audio;
  if (!activity.active) {
    return output;
  }

  AcousticTrack track;
  track.track_id = tracker_.assign(frame.stamp_ms, azimuth);
  track.stamp_ms = frame.stamp_ms;
  track.voice_activity = activity.active;
  track.speech_probability = activity.probability;
  track.clarity = clarity;
  track.azimuth_deg = azimuth;
  track.overlap_probability = estimateOverlap(phase_safe_frame);
  track.self_echo_probability = self_echo;
  track.features = extractFeatures(target_audio, frame.sample_rate_hz);

  output.target_track_id = track.track_id;
  output.tracks.push_back(track);
  return output;
}

std::int64_t nowMs()
{
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

}  // namespace voice_detection
