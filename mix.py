"""
Audio Mixing Module

This module provides professional audio mixing capabilities for combining
speech, background music, and sound effects into final audio productions.
"""

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

from config import get_config
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Get configuration
_config = get_config()


class AudioCache:
    """
    Audio cache helper class for pre-loading and caching audio files to avoid repeated IO operations.
    """
    def __init__(self):
        self.cache = {} 

    def get_audio(self, path: str) -> AudioSegment:
        if path not in self.cache:
            if os.path.exists(path):
                self.cache[path] = AudioSegment.from_wav(path)
            else:
                print(f"Warning: File missing: {path}, using 0.5s silent placeholder")
                self.cache[path] = AudioSegment.silent(duration=500)
        return self.cache[path]

    def get_duration(self, path: str) -> int:
        return len(self.get_audio(path))


class AudioMixer:
    """
    Core audio mixing module for professional audio production.

    Handles multi-track audio mixing with precise timing, volume balancing,
    and fade effects for speech, background music, and sound effects.
    """

    def __init__(self, 
                 script_path: str, 
                 output_dir: str, 
                 resource_dirs: Dict[str, str], 
                 gap_ms: int = None):
        """
        Initialize the audio mixer.
        
        Args:
            script_path: Path to JSONL script file
            output_dir: Output directory for results
            resource_dirs: Resource directories dict, must contain 'speech', 'sfx', 'bgm' keys
            gap_ms: Silent gap between sentences in milliseconds (uses default if None)
        """
        self.script_path = script_path
        self.output_dir = output_dir
        self.dirs = resource_dirs
        self.gap_ms = gap_ms or _config.audio.default_gap_ms
        self.loader = AudioCache()
        
    def _db_to_gain(self, db: float) -> float:
        return 10 ** (db / 20)

    def run(self) -> Dict[str, str]:
        """
        Execute the mixing process.
        
        Returns:
            Dict: Dictionary containing paths to generated files
        """
        # 1. Load data
        print(f"[Mixer] Loading script: {self.script_path}")
        data = []
        with open(self.script_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))

        # 2. Classify data
        speech_items = {}
        sfx_before = {}
        sfx_after = {}
        sfx_overlap = {}
        sfx_global = []
        bgm_events = []
        max_seq = 0

        for item in data:
            atype = item['audio_type']
            if atype == 'speech':
                seq = item['speech_seq']
                speech_items[seq] = item
                max_seq = max(max_seq, seq)
            elif atype == 'sfx':
                if 'scope_speech_seqs' in item:
                    sfx_global.append(item)
                else:
                    anchor = item['anchor_to_speech_seq']
                    place = item.get('placement', 'overlap')
                    if place == 'before': sfx_before.setdefault(anchor, []).append(item)
                    elif place == 'after': sfx_after.setdefault(anchor, []).append(item)
                    else: sfx_overlap.setdefault(anchor, []).append(item)
            elif atype == 'bgm':
                bgm_events.append(item)

        # 3. Calculate physical timeline
        print("[Mixer] Calculating physical timeline layout...")
        timeline_registry = {}
        current_cursor = 0

        for seq in range(1, max_seq + 1):
            if seq not in speech_items: continue
            
            # A. Before
            dur_before = 0
            if seq in sfx_before:
                for sfx in sfx_before[seq]:
                    path = os.path.join(self.dirs['sfx'], f"sfx_{sfx['clip_id']}.wav")
                    dur = self.loader.get_duration(path)
                    if 'duration' in sfx: dur = min(dur, int(sfx['duration'] * 1000))
                    dur_before = max(dur_before, dur)

            # B. Speech
            speech_path = os.path.join(self.dirs['speech'], f"speech_{seq}.wav")
            dur_speech = self.loader.get_duration(speech_path)

            # C. After
            dur_after = 0
            if seq in sfx_after:
                for sfx in sfx_after[seq]:
                    path = os.path.join(self.dirs['sfx'], f"sfx_{sfx['clip_id']}.wav")
                    dur = self.loader.get_duration(path)
                    if 'duration' in sfx: dur = min(dur, int(sfx['duration'] * 1000))
                    dur_after = max(dur_after, dur)

            # D. Register coordinates
            block_start = current_cursor
            speech_start = block_start + dur_before
            speech_end = speech_start + dur_speech
            sfx_after_end = speech_end + dur_after
            block_end = sfx_after_end + self.gap_ms

            timeline_registry[seq] = {
                'block_start': block_start,
                'speech_start': speech_start,
                'speech_end': speech_end,
                'sfx_after_start': speech_end,
                'block_end': block_end
            }
            current_cursor = block_end

        total_duration_ms = current_cursor

        # 4. Render tracks
        print("[Mixer] Starting track rendering...")
        track_speech = AudioSegment.silent(duration=total_duration_ms)
        track_sfx = AudioSegment.silent(duration=total_duration_ms)
        track_bgm = AudioSegment.silent(duration=total_duration_ms)

        # --- Track 1: Speech ---
        for seq, coords in timeline_registry.items():
            path = os.path.join(self.dirs['speech'], f"speech_{seq}.wav")
            seg = self.loader.get_audio(path)
            track_speech = track_speech.overlay(seg, position=coords['speech_start'])

        # --- Track 2: SFX ---
        # Before
        for seq, items in sfx_before.items():
            if seq not in timeline_registry: continue
            for item in items:
                path = os.path.join(self.dirs['sfx'], f"sfx_{item['clip_id']}.wav")
                seg = self.loader.get_audio(path) + item.get('vol', 0)
                if 'duration' in item: seg = seg[:int(item['duration']*1000)]
                track_sfx = track_sfx.overlay(seg, position=timeline_registry[seq]['block_start'])
        
        # After
        for seq, items in sfx_after.items():
            if seq not in timeline_registry: continue
            for item in items:
                path = os.path.join(self.dirs['sfx'], f"sfx_{item['clip_id']}.wav")
                seg = self.loader.get_audio(path) + item.get('vol', 0)
                if 'duration' in item: seg = seg[:int(item['duration']*1000)]
                track_sfx = track_sfx.overlay(seg, position=timeline_registry[seq]['sfx_after_start'])

        # Overlap
        for seq, items in sfx_overlap.items():
            if seq not in timeline_registry: continue
            for item in items:
                path = os.path.join(self.dirs['sfx'], f"sfx_{item['clip_id']}.wav")
                seg = self.loader.get_audio(path) + item.get('vol', 0)
                if 'duration' in item: seg = seg[:int(item['duration']*1000)]
                abs_pos = timeline_registry[seq]['speech_start'] + int(item.get('offset', 0) * 1000)
                track_sfx = track_sfx.overlay(seg, position=abs_pos)

        # Global Scope
        for item in sfx_global:
            scope = item['scope_speech_seqs']
            if not scope: continue
            start_seq, end_seq = min(scope), max(scope)
            if start_seq in timeline_registry and end_seq in timeline_registry:
                t_start = timeline_registry[start_seq]['block_start']
                t_end = timeline_registry[end_seq]['block_end']
                dur_req = t_end - t_start
                path = os.path.join(self.dirs['sfx'], f"sfx_{item['clip_id']}.wav")
                raw = self.loader.get_audio(path) + item.get('vol', 0)
                looped = raw * (math.ceil(dur_req / len(raw)) + 1)
                looped = looped[:dur_req].fade_in(500).fade_out(500)
                track_sfx = track_sfx.overlay(looped, position=t_start)

        # --- Track 3: BGM ---
        bgm_events.sort(key=lambda x: x['anchor_to_speech_seq'])
        bgm_groups = {}
        for ev in bgm_events: bgm_groups.setdefault(ev.get('bgm_seq', 0), []).append(ev)

        for b_seq, events in bgm_groups.items():
            start_ev = next((x for x in events if x['action'] == 'start'), None)
            if not start_ev: continue
            stop_ev = next((x for x in events if x['action'] == 'stop'), None)
            start_anchor = start_ev['anchor_to_speech_seq']
            if start_anchor not in timeline_registry: continue
            
            bgm_start_ms = timeline_registry[start_anchor]['block_start']
            
            if stop_ev and stop_ev['anchor_to_speech_seq'] in timeline_registry:
                bgm_end_ms = timeline_registry[stop_ev['anchor_to_speech_seq']]['block_end']
            else:
                bgm_end_ms = total_duration_ms
            
            dur_req = bgm_end_ms - bgm_start_ms
            if dur_req > 0:
                path = os.path.join(self.dirs['bgm'], f"bgm_{start_ev['clip_id']}.wav")
                raw = self.loader.get_audio(path) + start_ev.get('vol', 0)
                looped = raw * (math.ceil(dur_req / len(raw)) + 1)
                looped = looped[:dur_req].fade_in(2000).fade_out(2000)
                track_bgm = track_bgm.overlay(looped, position=bgm_start_ms)

        # 5. Export
        print("[Mixer] Saving files...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        paths = {
            "speech": os.path.join(self.output_dir, "track_1_speech.wav"),
            "sfx": os.path.join(self.output_dir, "track_2_sfx.wav"),
            "bgm": os.path.join(self.output_dir, "track_3_bgm.wav"),
            "master": os.path.join(self.output_dir, "final_mix_master.wav")
        }

        track_speech.export(paths["speech"], format="wav")
        track_sfx.export(paths["sfx"], format="wav")
        track_bgm.export(paths["bgm"], format="wav")
        
        final_mix = track_speech.overlay(track_bgm).overlay(track_sfx)
        final_mix.export(paths["master"], format="wav")
        
        print("[Mixer] Task completed!")
        return paths

# For compatibility, keep a simple entry point when running this script directly
if __name__ == "__main__":
    # Define default parameters here
    mixer = AudioMixer(
        script_path="4.jsonl",
        output_dir="output",
        resource_dirs={
            "speech": "generated_audio/speech",
            "sfx": "generated_audio/cache",
            "bgm": "generated_audio/cache"
        }
    )
    mixer.run()