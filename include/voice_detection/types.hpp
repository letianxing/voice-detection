#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace voice_detection
{

struct MicrophoneProfile
{
  std::string id;
  std::string label;
  int sample_rate_hz{48000};
  int input_channels{1};
  std::vector<int> doa_channel_indices;
  std::vector<std::array<float, 3>> mic_positions_m;
  int preferred_block_ms{20};
  bool supports_doa{false};
  bool supports_raw_multichannel{false};
  std::optional<int> hardware_beam_channel_index;
  std::optional<int> reference_channel_index;
  std::string notes;
};

struct AudioFrame
{
  std::vector<float> samples;
  int sample_rate_hz{48000};
  int channels{1};
  std::int64_t stamp_ms{0};

  [[nodiscard]] std::size_t frames() const
  {
    return channels <= 0 ? 0 : samples.size() / static_cast<std::size_t>(channels);
  }

  [[nodiscard]] float at(std::size_t frame, int channel) const
  {
    return samples[(frame * static_cast<std::size_t>(channels)) + static_cast<std::size_t>(channel)];
  }
};

struct VoiceActivity
{
  bool active{false};
  float probability{0.0F};
  float rms_dbfs{-120.0F};
  float noise_floor_dbfs{-65.0F};
  float snr_db{0.0F};
};

struct AudioFeatures
{
  float zcr{0.0F};
  float rms{0.0F};
  float pitch{0.0F};
  float hnr{0.0F};
  std::array<float, 12> mfcc{};
};

struct SpeechTranscript
{
  std::string track_id;
  std::string text;
  bool is_final{false};
  std::string language{"unknown"};
  float clarity{0.0F};
  float confidence{0.0F};
  std::optional<std::int64_t> started_ms;
  std::optional<std::int64_t> ended_ms;
  std::optional<std::int64_t> emitted_ms;
};

struct AcousticTrack
{
  std::string track_id;
  std::int64_t stamp_ms{0};
  bool voice_activity{false};
  float speech_probability{0.0F};
  float clarity{0.0F};
  std::optional<float> azimuth_deg;
  std::optional<float> elevation_deg;
  std::optional<float> distance_m;
  float overlap_probability{0.0F};
  float self_echo_probability{0.0F};
  std::string speaker_label{"unknown"};
  AudioFeatures features;
  std::optional<SpeechTranscript> transcript;
};

struct FrontendOutput
{
  std::int64_t stamp_ms{0};
  std::vector<AcousticTrack> tracks;
  std::vector<float> target_audio;
  std::string target_track_id;
};

}  // namespace voice_detection
