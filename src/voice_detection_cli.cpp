#include "voice_detection/frontend.hpp"
#include "voice_detection/profiles.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
#include <string>

namespace
{

voice_detection::AudioFrame syntheticFrame(
  const voice_detection::MicrophoneProfile & profile,
  float azimuth_deg)
{
  constexpr float speed = 343.0F;
  const int sample_count = static_cast<int>(profile.sample_rate_hz * 0.12F);
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.0F, 0.04F);
  std::vector<float> base(static_cast<std::size_t>(sample_count), 0.0F);
  for (int i = 0; i < sample_count; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(profile.sample_rate_hz);
    base[static_cast<std::size_t>(i)] =
      (0.14F * std::sin(2.0F * static_cast<float>(M_PI) * 220.0F * t)) + noise(rng);
  }

  voice_detection::AudioFrame frame;
  frame.sample_rate_hz = profile.sample_rate_hz;
  frame.channels = profile.input_channels;
  frame.stamp_ms = voice_detection::nowMs();
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

void printProfiles()
{
  for (const auto & profile : voice_detection::defaultProfiles()) {
    std::cout << profile.id << "\t" << profile.label << "\tchannels=" << profile.input_channels
              << "\tdoa=" << (profile.supports_doa ? "true" : "false") << '\n';
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  const std::string command = argc > 1 ? argv[1] : "help";
  if (command == "profiles") {
    printProfiles();
    return 0;
  }
  if (command == "simulate") {
    const std::string profile_id = argc > 2 ? argv[2] : "sipeed_6_plus_1_usb_array";
    const float azimuth = argc > 3 ? std::stof(argv[3]) : 30.0F;
    const auto profile = voice_detection::getDefaultProfile(profile_id);
    auto frame = syntheticFrame(profile, azimuth);
    voice_detection::CocktailFrontend frontend(profile);
    const auto output = frontend.process(frame);
    for (const auto & track : output.tracks) {
      std::cout << "{"
                << "\"track_id\":\"" << track.track_id << "\","
                << "\"voice_activity\":" << (track.voice_activity ? "true" : "false") << ","
                << "\"speech_probability\":" << track.speech_probability << ","
                << "\"clarity\":" << track.clarity << ",";
      if (track.azimuth_deg.has_value()) {
        std::cout << "\"azimuth_deg\":" << *track.azimuth_deg << ",";
      } else {
        std::cout << "\"azimuth_deg\":null,";
      }
      std::cout << "\"overlap_probability\":" << track.overlap_probability << "}" << '\n';
    }
    return 0;
  }

  std::cerr << "usage: voice_detection_cli profiles | simulate [profile] [azimuth_deg]\n";
  return command == "help" ? 0 : 2;
}
