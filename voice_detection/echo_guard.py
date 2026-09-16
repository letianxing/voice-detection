"""Conservative attribution of ASR fragments matching contemporaneous robot playback."""
import re
from difflib import SequenceMatcher


def normalized(text):
    return re.sub(r'[^\w\u4e00-\u9fff]', '', str(text).lower())


def match_playback(text, started_ms, ended_ms, records, speaker_similarity=0.):
    candidate = normalized(text)
    if len(candidate) < 4 or speaker_similarity >= .75:
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
        score = 1. if candidate in reference else sum(block.size for block in SequenceMatcher(None,candidate,reference,autojunk=False).get_matching_blocks()) / len(candidate)
        if score >= .82:
            return {'playback_id': record.get('id'), 'turn_id': record.get('turn_id'), 'score': round(score,3),
                    'reason': 'matches_contemporaneous_robot_playback'}
    return None
