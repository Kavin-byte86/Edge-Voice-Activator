#!/usr/bin/env python3
"""
GroundWatch Hardware Test Suite
================================
Tests the ESP32 KWS system via serial port.

Test 1: RMS mic sanity check  - verifies mic is live and picking up sound
Test 2: Play a training WAV   - plays real gTTS sample near mic to confirm model triggers
Test 3: Confidence probe      - monitors raw confidence values in real time
Test 4: Threshold sweep       - tells you what confidence YOUR voice achieves

Usage:
  py -3.10 scripts/hardware_test.py --port COM16
"""

import sys
import os
import time
import argparse
import threading
import queue

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
BAUD      = 115200
RMS_LIVE  = 500    # RMS above this = mic is picking up sound
RMS_VOICE = 3000   # RMS above this = voice-level signal (you need this when speaking)

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_rms(line: str):
    """Extract RMS value from [IDLE] lines."""
    if "RMS:" in line:
        try:
            rms_part = line.split("RMS:")[1].strip().split()[0]
            return int(rms_part)
        except Exception:
            pass
    return None

def parse_confidence(line: str):
    """Extract confidence from detection lines."""
    if "Confidence:" in line and "Feat:" in line:
        try:
            conf = float(line.split("Confidence:")[1].strip().split()[0])
            return conf
        except Exception:
            pass
    return None

def parse_keyword_confidence(line: str):
    """Extract confidence from benchmark block."""
    if "Keyword Confidence:" in line:
        try:
            return float(line.strip().split()[-1])
        except Exception:
            pass
    return None


class SerialReader:
    """Non-blocking serial reader that feeds lines into a queue."""
    def __init__(self, port, baud):
        self.ser  = serial.Serial(port, baud, timeout=1)
        self.q    = queue.Queue()
        self._run = True
        self._t   = threading.Thread(target=self._read_loop, daemon=True)
        self._t.start()

    def _read_loop(self):
        while self._run:
            try:
                raw = self.ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    self.q.put(line)
            except Exception:
                pass

    def get_lines(self, timeout=0.1):
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                lines.append(self.q.get_nowait())
            except queue.Empty:
                time.sleep(0.01)
        return lines

    def close(self):
        self._run = False
        self.ser.close()


# ── Test 1: Mic sanity ────────────────────────────────────────────────────────
def test_mic_sanity(reader, duration=5):
    print("\n" + "="*55)
    print("TEST 1: Mic Sanity Check")
    print("="*55)
    print(f"  Listening for {duration}s. Keep the room quiet.")
    print(f"  Expected idle RMS: 100-500")
    print(f"  RMS when you speak: should be > {RMS_VOICE}\n")

    rms_values = []
    t_end = time.time() + duration
    while time.time() < t_end:
        for line in reader.get_lines(0.2):
            rms = parse_rms(line)
            if rms is not None:
                rms_values.append(rms)
                bar = "#" * min(50, rms // 100)
                status = "QUIET" if rms < RMS_LIVE else ("VOICE" if rms > RMS_VOICE else "MIC_LIVE")
                print(f"  RMS={rms:6d}  [{bar:<50}]  {status}")

    if rms_values:
        avg = sum(rms_values) / len(rms_values)
        mx  = max(rms_values)
        print(f"\n  Avg RMS: {avg:.0f}  |  Peak RMS: {mx}")
        if avg < RMS_LIVE:
            print("  RESULT: WARNING - avg RMS very low. Check mic wiring.")
        elif avg < 200:
            print("  RESULT: PASS - mic is live but quiet room.")
        else:
            print("  RESULT: PASS - mic receiving ambient signal.")
    else:
        print("  RESULT: FAIL - no [IDLE] lines received. Is firmware running?")


# ── Test 2: Play training WAV ─────────────────────────────────────────────────
def test_play_training_sample(reader, wav_dir):
    print("\n" + "="*55)
    print("TEST 2: Play Training Sample Near Mic")
    print("="*55)

    pos_dir = os.path.join(wav_dir, "dataset", "raw", "positive")
    if not os.path.isdir(pos_dir):
        print(f"  SKIP: Positive sample dir not found: {pos_dir}")
        return

    # Pick 3 samples
    samples = sorted(os.listdir(pos_dir))[:5]
    if not samples:
        print("  SKIP: No positive samples found.")
        return

    print(f"  Found {len(os.listdir(pos_dir))} positive samples.")
    print("  Will play 5 samples through your PC speakers.")
    print("  Hold the INMP441 mic CLOSE to your speaker (within 10cm).\n")

    try:
        import pygame
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=512)
    except ImportError:
        print("  pygame not found. Trying winsound (Windows)...")
        try:
            import winsound
            for fname in samples:
                fpath = os.path.join(pos_dir, fname)
                print(f"  Playing: {fname}")
                print("  >> Watch for [KEYWORD DETECTED] on serial <<")
                winsound.PlaySound(fpath, winsound.SND_FILENAME)
                time.sleep(0.5)
                # Check for detection
                for line in reader.get_lines(1.5):
                    print("    " + line)
                    if "KEYWORD DETECTED" in line:
                        print("  *** DETECTED! Model works with training audio. ***")
            return
        except Exception as e:
            print(f"  winsound failed: {e}")
            print("  MANUAL TEST: Open any of these WAV files in Windows Media Player")
            print("  and hold the mic close to your speaker:")
            for s in samples:
                print(f"    {os.path.join(pos_dir, s)}")
            return

    for fname in samples:
        fpath = os.path.join(pos_dir, fname)
        print(f"  Playing: {fname}")
        print("  >> Watch for [KEYWORD DETECTED] on serial <<")
        try:
            pygame.mixer.music.load(fpath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            print(f"    Error: {e}")

        # Check 2 seconds for detection
        detected = False
        for line in reader.get_lines(2.0):
            if line.strip():
                print("    SERIAL: " + line)
            if "KEYWORD DETECTED" in line:
                detected = True
        if detected:
            print("  *** DETECTED! gTTS audio triggers the model correctly. ***")
        else:
            print("  No detection for this sample.")
        time.sleep(0.3)


# ── Test 3: Live confidence probe ─────────────────────────────────────────────
def test_live_confidence(reader, duration=20):
    print("\n" + "="*55)
    print("TEST 3: Live Confidence Probe")
    print("="*55)
    print(f"  Watching for {duration}s. Say 'Akash Go' repeatedly.")
    print("  You'll see the raw confidence each inference cycle.")
    print("  Target: confidence > 0.25 to trigger\n")

    detections = []
    t_end = time.time() + duration
    prev_rms = 0

    while time.time() < t_end:
        for line in reader.get_lines(0.1):
            rms = parse_rms(line)
            conf = parse_confidence(line)

            if rms is not None:
                # Only print when RMS spikes (speech detected)
                if rms > RMS_VOICE and rms > prev_rms * 1.5:
                    print(f"  [MIC SPIKE] RMS={rms}  <- speak now!")
                prev_rms = rms

            if conf is not None:
                detections.append(conf)
                bar = "#" * int(conf * 40)
                marker = " *** TRIGGERED ***" if conf >= 0.25 else ""
                print(f"  Confidence: {conf:.4f}  [{bar:<40}]{marker}")

            if "KEYWORD DETECTED" in line:
                print(f"\n  >>> DETECTION EVENT! <<<\n")

    if detections:
        print(f"\n  Total detection events: {len(detections)}")
        print(f"  Max confidence seen:    {max(detections):.4f}")
        print(f"  Avg confidence seen:    {sum(detections)/len(detections):.4f}")
        if max(detections) < 0.25:
            print("\n  DIAGNOSIS: Max confidence never reached 0.25.")
            print("  -> Your voice does not match the gTTS training data well.")
            print("  -> SOLUTION: Record your own voice and retrain.")
        elif max(detections) < 0.50:
            print("\n  DIAGNOSIS: Model detects weakly. Retrain with real voice data.")
        else:
            print("\n  DIAGNOSIS: Model responds to your voice. Tune threshold.")
    else:
        print("  No confidence events captured.")


# ── Test 4: RMS voice level guide ─────────────────────────────────────────────
def test_voice_level_guide(reader, duration=15):
    print("\n" + "="*55)
    print("TEST 4: Voice Level Guide")
    print("="*55)
    print("  This test tells you how loud your voice registers on the mic.")
    print("  Say 'Akash Go' at different distances/volumes.\n")
    print("  Distance guide:")
    print("    5cm  from mic  → RMS should be > 8000")
    print("    10cm from mic  → RMS should be > 4000")
    print("    30cm from mic  → RMS should be > 1500")
    print("    Silence        → RMS should be < 500\n")

    t_end = time.time() + duration
    while time.time() < t_end:
        for line in reader.get_lines(0.1):
            rms = parse_rms(line)
            if rms is not None:
                if rms < 200:
                    level = "SILENT       "
                elif rms < 500:
                    level = "ambient noise"
                elif rms < 1500:
                    level = "quiet room   "
                elif rms < 3000:
                    level = "soft voice   "
                elif rms < 6000:
                    level = "NORMAL VOICE "
                elif rms < 12000:
                    level = "LOUD VOICE   "
                else:
                    level = "VERY LOUD    "
                bar = "#" * min(60, rms // 200)
                print(f"  {level}  RMS={rms:6d}  [{bar:<60}]")
            if "KEYWORD DETECTED" in line:
                print("  *** DETECTED ***")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   default="COM16",   help="Serial port (default: COM16)")
    parser.add_argument("--test",   type=int, default=0, help="Run specific test (1-4). 0=all")
    parser.add_argument("--wav-dir", default=None,     help="Project root for WAV samples")
    args = parser.parse_args()

    # Auto-detect project root
    wav_dir = args.wav_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print("\n  GroundWatch Hardware Test Suite")
    print(f"  Port: {args.port}  |  Project: {wav_dir}\n")

    try:
        reader = SerialReader(args.port, BAUD)
        print(f"  Connected to {args.port} @ {BAUD} baud.")
        time.sleep(0.5)  # flush startup noise
    except serial.SerialException as e:
        print(f"\n  ERROR: Cannot open {args.port}: {e}")
        print("  Is the serial monitor still running? Close it first.")
        sys.exit(1)

    try:
        if args.test == 0 or args.test == 1:
            test_mic_sanity(reader, duration=5)
        if args.test == 0 or args.test == 2:
            test_play_training_sample(reader, wav_dir)
        if args.test == 0 or args.test == 3:
            test_live_confidence(reader, duration=20)
        if args.test == 0 or args.test == 4:
            test_voice_level_guide(reader, duration=15)
    finally:
        reader.close()
        print("\n  Tests complete.")


if __name__ == "__main__":
    main()
