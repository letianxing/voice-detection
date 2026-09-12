#include "voice_detection/frontend.hpp"
#include "voice_detection/profiles.hpp"
#include "voice_detection/signal_enhancement.hpp"
#include "voice_detection/tracker.hpp"
#include "voice_detection/vad.hpp"

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

namespace
{

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

voice_detection::AudioFrame syntheticArrayFrame(
  const voice_detection::MicrophoneProfile & profile,
  float azimuth_deg)
{
  constexpr float speed = 343.0F;
  const int sample_count = static_cast<int>(profile.sample_rate_hz * 0.12F);
  std::mt19937 rng(11);
  std::normal_distribution<float> noise(0.0F, 0.03F);
  std::vector<float> base(static_cast<std::size_t>(sample_count), 0.0F);
  for (int i = 0; i < sample_count; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(profile.sample_rate_hz);
    base[static_cast<std::size_t>(i)] =
      (0.14F * std::sin(2.0F * static_cast<float>(M_PI) * 260.0F * t)) + noise(rng);
  }

  voice_detection::AudioFrame frame;
  frame.sample_rate_hz = profile.sample_rate_hz;
  frame.channels = profile.input_channels;
  frame.stamp_ms = 1000;
  frame.samples.assign(static_cast<std::size_t>(sample_count * profile.input_channels), 0.0F);

  const float rad = azimuth_deg * static_cast<float>(M_PI) / 180.0F;
  const std::array<float, 3> direction{std::cos(rad), std::sin(rad), 0.0F};
  std::vector<float> arrivals;
  for (int channel = 0; channel < profile.input_channels; ++channel) {
    std::array<float, 3> pos{0.0F, 0.0F, 0.0F};
    if (static_cast<std::size_t>(channel) < profile.mic_positions_m.size()) {
      pos = profile.mic_positions_m[static_cast<std::size_t>(channel)];
    }
    arrivals.push_back(-((pos[0] * direction[0]) + (pos[1] * direction[1]) + (pos[2] * direction[2])) / speed);
  }
  const float min_arrival = *std::min_element(arrivals.begin(), arrivals.end());
  for (int channel = 0; channel < profile.input_channels; ++channel) {
    const int shift = static_cast<int>(std::lround((arrivals[static_cast<std::size_t>(channel)] - min_arrival) *
      profile.sample_rate_hz));
    for (int i = shift; i < sample_count; ++i) {
      frame.samples[(static_cast<std::size_t>(i) * static_cast<std::size_t>(profile.input_channels)) +
        static_cast<std::size_t>(channel)] = base[static_cast<std::size_t>(i - shift)];
    }
  }
  return frame;
}

}  // namespace

int main()
{
  auto profiles = voice_detection::defaultProfiles();
  require(!profiles.empty(), "profiles should be present");

  const auto mac = voice_detection::getDefaultProfile("mac_builtin");
  voice_detection::AudioFrame mono;
  mono.sample_rate_hz = mac.sample_rate_hz;
  mono.channels = 1;
  mono.stamp_ms = 100;
  mono.samples.assign(4800, 0.0F);
  for (std::size_t i = 0; i < mono.samples.size(); ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(mono.sample_rate_hz);
    mono.samples[i] = 0.08F * std::sin(2.0F * static_cast<float>(M_PI) * 220.0F * t);
  }
  voice_detection::CocktailFrontend mac_frontend(mac);
  const auto mono_output = mac_frontend.process(mono);
  require(mono_output.tracks.size() == 1, "mono frame should produce one voice track");
  require(mono_output.tracks.front().track_id == "voice_mono", "mono track id should be stable");
  require(!mono_output.tracks.front().azimuth_deg.has_value(), "mono profile should not estimate azimuth");

  const auto sipeed = voice_detection::getDefaultProfile("sipeed_6_plus_1_usb_array");
  voice_detection::CocktailFrontend array_frontend(sipeed);
  const auto array_output = array_frontend.process(syntheticArrayFrame(sipeed, 30.0F));
  require(array_output.tracks.size() == 1, "array frame should produce one track");
  require(array_output.tracks.front().azimuth_deg.has_value(), "array should estimate azimuth");
  require(
    std::abs(*array_output.tracks.front().azimuth_deg - 30.0F) < 24.0F,
    "array estimated azimuth should be near synthetic source");

  voice_detection::SourceTracker tracker;
  const auto first = tracker.assign(1000, 10.0F);
  const auto second = tracker.assign(1020, 18.0F);
  const auto third = tracker.assign(1050, 80.0F);
  require(first == second, "close azimuths should reuse voice id");
  require(first != third, "far azimuth should create new voice id");

  std::vector<float> playback(960, 0.0F);
  std::vector<float> microphone(960, 0.0F);
  for (std::size_t i = 0; i < playback.size(); ++i) {
    playback[i] = 0.2F * std::sin(static_cast<float>(i) * 0.1F);
    microphone[i] = playback[i] * 0.8F;
  }
  voice_detection::LinearEchoCanceller echo_canceller(0.0F);
  const auto cancelled = echo_canceller.process(microphone, playback);
  double input_energy = 0.0;
  double output_energy = 0.0;
  for (std::size_t i = 0; i < microphone.size(); ++i) {
    input_energy += microphone[i] * microphone[i];
    output_energy += cancelled[i] * cancelled[i];
  }
  require(output_energy < input_energy * 0.01, "aligned playback reference should reduce echo");
  return 0;
}
