#!/usr/bin/env python3
"""
Time dilation for faster-than-real-time training.

The robot's modules are unmodified and wall-clock driven: field_agent
ticks at 5Hz via time.sleep(0.2), world_state publishes at 2Hz, timers
compare time.time() deltas, etc. To run an episode at N x real speed
*faithfully* - so every tick, sensor update and timer keeps the same
relative cadence, just compressed - we dilate the clock the whole system
reads, in lockstep across every process.

install() (called by run_module in each module subprocess) rebinds
time.sleep / time.time / time.monotonic / time.perf_counter so that:

  - sleeps return N x sooner  -> tick loops run N x faster
  - time.time()/monotonic()   advance N x faster, sharing a single epoch
    with the launcher (passed via env) so cross-process timestamps stay
    comparable - CLOCK_REALTIME and CLOCK_MONOTONIC are system-wide, so
    every process computes the identical dilated value for a real instant

The launcher side (physics dt, sensor timestamps, episode clock) uses
sim_now()/scale from here without monkeypatching, keeping its own loop
pacing real. Threading's internally-cached monotonic is intentionally
left real: lock timeouts are transport plumbing, not simulation time.

Fidelity note: this is exact only to the extent module behavior is a
pure function of its clock. It is; but subprocess/OS scheduling is not
infinitely fast, so very large N eventually starves the tick loops of
real CPU time. See run_training.py --speedf help for the practical range.
"""
import os
import time as _time

_installed = False


def speedf():
    try:
        return max(1.0, float(os.environ.get("SIM_SPEEDF", "1") or "1"))
    except ValueError:
        return 1.0


def make_clock(scale, t0, m0):
    """Return (dilated_time, dilated_monotonic) sharing the given epochs."""
    real_time, real_mono = _time.time, _time.monotonic
    return (lambda: t0 + (real_time() - t0) * scale,
            lambda: m0 + (real_mono() - m0) * scale)


def install():
    """Dilate this process's clock per SIM_SPEEDF/SIM_CLOCK_T0/M0 env.

    Idempotent and a no-op at 1x. Returns the active speed factor."""
    global _installed
    scale = speedf()
    if _installed or scale == 1.0:
        return scale
    t0 = float(os.environ.get("SIM_CLOCK_T0", _time.time()))
    m0 = float(os.environ.get("SIM_CLOCK_M0", _time.monotonic()))
    dilated_time, dilated_mono = make_clock(scale, t0, m0)
    real_sleep, real_perf = _time.sleep, _time.perf_counter
    perf0 = real_perf()

    def dilated_sleep(seconds):
        if seconds and seconds > 0:
            real_sleep(seconds / scale)

    _time.time = dilated_time
    _time.monotonic = dilated_mono
    _time.perf_counter = lambda: (real_perf() - perf0) * scale
    _time.sleep = dilated_sleep
    _installed = True
    return scale
