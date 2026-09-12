#include "voice_detection/profiles.hpp"

#include <stdexcept>

namespace voice_detection
{

std::vector<MicrophoneProfile> defaultProfiles()
{
  MicrophoneProfile mac;
  mac.id = "mac_builtin";
  mac.label = "Mac built-in microphone";
  mac.sample_rate_hz = 48000;
  mac.input_channels = 1;
  mac.preferred_block_ms = 20;
  mac.supports_doa = false;
  mac.supports_raw_multichannel = false;
  mac.notes = "Good for VAD/ASR and local testing; not enough for reliable cocktail-party DOA.";

  MicrophoneProfile win;
  win.id = "windows_wasapi_builtin";
  win.label = "Windows WASAPI default microphone";
  win.sample_rate_hz = 48000;
  win.input_channels = 1;
  win.preferred_block_ms = 20;
  win.supports_doa = false;
  win.supports_raw_multichannel = false;
  win.notes = "Use for single-user VAD/ASR smoke tests.";

  MicrophoneProfile sipeed;
  sipeed.id = "sipeed_6_plus_1_usb_array";
  sipeed.label = "Sipeed 6+1 Mic Array USB";
  sipeed.sample_rate_hz = 48000;
  sipeed.input_channels = 8;
  sipeed.doa_channel_indices = {0, 1, 2, 3, 4, 5};
  sipeed.mic_positions_m = {
    std::array<float, 3>{0.035F, 0.0F, 0.0F},
    std::array<float, 3>{0.0175F, 0.03031F, 0.0F},
    std::array<float, 3>{-0.0175F, 0.03031F, 0.0F},
    std::array<float, 3>{-0.035F, 0.0F, 0.0F},
    std::array<float, 3>{-0.0175F, -0.03031F, 0.0F},
    std::array<float, 3>{0.0175F, -0.03031F, 0.0F},
    std::array<float, 3>{0.0F, 0.0F, 0.0F},
    std::array<float, 3>{0.0F, 0.0F, 0.0F}};
  sipeed.preferred_block_ms = 20;
  sipeed.supports_doa = true;
  sipeed.supports_raw_multichannel = true;
  sipeed.hardware_beam_channel_index = 6;
  sipeed.reference_channel_index = 7;
  sipeed.notes =
    "MA-USB8 8ch: CH0-CH5 raw outer ring PCM, CH6 board beam/average output, "
    "CH7 raw/reference candidate. Verify channel order; geometry assumes a 35 mm ring.";

  MicrophoneProfile reachy;
  reachy.id = "reachy_mini_native";
  reachy.label = "Reachy Mini native audio";
  reachy.sample_rate_hz = 48000;
  reachy.input_channels = 4;
  reachy.preferred_block_ms = 20;
  reachy.supports_doa = true;
  reachy.supports_raw_multichannel = true;
  reachy.notes = "Use the official media stack when running on the robot.";

  return {mac, win, sipeed, reachy};
}

MicrophoneProfile getDefaultProfile(const std::string & id)
{
  for (const auto & profile : defaultProfiles()) {
    if (profile.id == id) {
      return profile;
    }
  }
  throw std::invalid_argument("unknown microphone profile: " + id);
}

}  // namespace voice_detection
