import unittest

import numpy as np

from voice_detection.streaming_audio import DelayedCrossfadeJoiner, PcmPacketizer


class StreamingAudioTest(unittest.TestCase):
    def test_joiner_delays_overlap_until_next_chunk(self):
        joiner = DelayedCrossfadeJoiner(overlap_samples=2)

        first = joiner.append(np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        second = joiner.append(np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        tail = joiner.flush()

        np.testing.assert_allclose(first, [1.0, 1.0])
        self.assertEqual(second.size, 2)
        np.testing.assert_allclose(tail, [0.0, 0.0])

    def test_packetizer_emits_fixed_pcm_packets_and_pads_tail(self):
        packetizer = PcmPacketizer(packet_samples=4, overlap_samples=0)

        packets = packetizer.push(np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32))
        packets.extend(packetizer.flush())

        self.assertEqual([packet.size for packet in packets], [4, 4])
        self.assertEqual(packets[0].dtype, np.int16)
        self.assertTrue(np.all(packets[1][1:] == 0))


if __name__ == "__main__":
    unittest.main()
