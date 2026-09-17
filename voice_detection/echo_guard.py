"""Conservative attribution of ASR fragments matching contemporaneous robot playback."""
import re
from difflib import SequenceMatcher


def normalized(text):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', str(text).lower())


# How much of the playback reference is still correlated with the microphone
# after echo cancellation. Measured over one field session on the array: every
# transcript of a real person came out at exactly 0.000 (five of them, spoken
# when nothing was playing), and every transcript that was the robot hearing
# itself came out between 0.457 and 0.758 (eight of them). A barge-in leaves
# mostly the person's own signal behind, which is what pushes this down.
ECHO_RESIDUAL = .45
# Similarity to what was played, when the residual already says it is us. The
# text bar alone has to be strict because it is the only evidence; with the
# residual corroborating it, a couple of misheard characters under the robot's
# own voice must not turn the echo into a person. In the field "我是小圆呀"
# came back as "我是小圆样" (0.80) and the weather reply came back with a
# mangled tail (0.77) — both just under the strict bar, both answered, and the
# robot then heard its answer and answered that.
CORROBORATED_TEXT = .55


def match_playback(text, started_ms, ended_ms, records, speaker_similarity=0., self_echo_probability=0.):
    candidate = normalized(text)
    if not candidate or speaker_similarity >= .75:
        return None
    echoing = float(self_echo_probability or 0) >= ECHO_RESIDUAL
    if len(candidate) < 4 and not echoing:
        # Short fragments are left alone unless the audio itself says this is
        # our own playback: "好的" is something people say.
        return None
    for record in reversed(records):
        start = record.get('started_ms') or 0
        end = record.get('ended_ms') or ended_ms
        if ended_ms < start or started_ms > end + 600:
            continue
        reference = normalized(record.get('text', ''))
        if not reference:
            continue
        overlap = max(0, min(ended_ms, end + 600) - max(started_ms, start))
        if overlap < .5 * max(1, ended_ms-started_ms):
            continue
        contained = candidate in reference
        score = 1. if contained else sum(block.size for block in SequenceMatcher(None,candidate,reference,autojunk=False).get_matching_blocks()) / len(candidate)
        if score >= .82 or (echoing and (score >= CORROBORATED_TEXT or contained)):
            return {'playback_id': record.get('id'), 'turn_id': record.get('turn_id'), 'score': round(score,3),
                    'echo_residual': round(float(self_echo_probability or 0), 3),
                    'reason': 'matches_contemporaneous_robot_playback' if score >= .82
                              else 'echo_residual_over_our_own_playback'}
    return None
