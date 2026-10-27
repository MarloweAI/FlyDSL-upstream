# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors


from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass

from ...expr.rocdl import ktrace_format as _format

_HIP_SUCCESS = 0
_RT = None


def _runtime():
    global _RT
    if _RT is None:
        from ...compiler.jit_executor import _resolve_runtime_libs

        _RT = ctypes.CDLL(_resolve_runtime_libs()[0])
        _RT.mgpuMemAlloc.restype = ctypes.c_void_p
        _RT.mgpuMemAlloc.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_bool]
        _RT.mgpuTraceBufferClear.restype = ctypes.c_int
        _RT.mgpuTraceBufferClear.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        _RT.mgpuTraceBufferRead.restype = ctypes.c_int
        _RT.mgpuTraceBufferRead.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    return _RT


def _check(code: int, what: str) -> None:
    if code != _HIP_SUCCESS:
        raise RuntimeError(f"{what} failed with HIP error {code}")


def _current_device() -> int:
    from . import get_device_runtime

    return get_device_runtime().current_device_id()


@dataclass
class TraceBuffer:

    device_ptr: int
    capacity_slots: int
    device_id: int = -1

    def synchronize(self) -> None:
        self.claimed_slots()

    def reset(self) -> None:
        self._require_own_device()
        _check(
            _runtime().mgpuTraceBufferClear(
                ctypes.c_void_p(self.device_ptr), (self.capacity_slots + 1) * _format.RECORD_BYTES
            ),
            "clearing the trace buffer",
        )

    def _require_own_device(self) -> None:
        if self.device_id < 0:
            return
        current = _current_device()
        if current != self.device_id:
            raise RuntimeError(
                f"ktrace: the trace buffer was allocated on device {self.device_id} but the "
                f"current device is {current}. Tracing spans one device per process; run the "
                "traced kernels on the device the buffer belongs to, or trace them in "
                "separate processes."
            )

    @property
    def partitions(self) -> int:
        return _format.partition_layout(self.capacity_slots)[0]

    @property
    def partition_slots(self) -> int:
        return _format.partition_layout(self.capacity_slots)[1]

    def partition_claims(self) -> list[int]:
        """Slots claimed in each partition since the last reset; a claim past ``partition_slots`` overflowed."""
        self._require_own_device()
        stride = _format.CURSOR_STRIDE_BYTES
        raw = (ctypes.c_char * (self.partitions * stride))()
        _check(
            _runtime().mgpuTraceBufferRead(raw, ctypes.c_void_p(self.device_ptr + _format.RECORD_BYTES), len(raw)),
            "reading the trace cursors",
        )
        return [struct.unpack_from("<Q", raw, p * stride)[0] for p in range(self.partitions)]

    def claimed_slots(self) -> int:
        return sum(self.partition_claims())

    def header(self) -> tuple[int, int, int]:
        """(waves dropped past the buffer, launches, waves of the launch) since the last reset.

        The first and last are written only by kernels compiled with static slots."""
        self._require_own_device()
        raw = (ctypes.c_char * 12)()
        _check(_runtime().mgpuTraceBufferRead(raw, ctypes.c_void_p(self.device_ptr), len(raw)), "reading the header")
        return struct.unpack_from("<III", raw)

    def _read_static(self, dropped: int, launches: int, waves: int) -> int:
        if launches > 1:
            raise RuntimeError(
                f"ktrace: static slots hold one launch per read, but {launches} launches ran since the last read. "
                "Read or clear after each launch, or use FLYDSL_KTRACE_SLOTS=partitioned."
            )
        if dropped:
            needed = _format.first_record_slot(_format.MAX_PARTITIONS) + waves * _format.EVENTS_PER_WAVE
            raise RuntimeError(
                f"trace overflowed: {dropped} of {waves} waves did not fit the buffer with static slots. "
                f"Re-run with FLYDSL_KTRACE_BUFFER_BYTES>={needed * _format.RECORD_BYTES} or a block filter."
            )
        return waves * _format.EVENTS_PER_WAVE * _format.RECORD_BYTES

    def launch_count(self) -> int:
        self._require_own_device()
        out = ctypes.c_uint32(0)
        _check(
            _runtime().mgpuTraceBufferRead(ctypes.byref(out), ctypes.c_void_p(self.device_ptr + 4), 4),
            "reading the trace launch count",
        )
        return int(out.value)

    def read_records(self) -> list[dict]:
        dropped, launches, static_waves = self.header()
        claims = self.partition_claims()
        size = self.partition_slots
        worst = max(claims, default=0)
        if static_waves and worst:
            raise RuntimeError(
                "ktrace: kernels with static and partitioned slots wrote the same trace between reads; "
                "use one FLYDSL_KTRACE_SLOTS setting per read."
            )
        if static_waves:
            nbytes = self._read_static(dropped, launches, static_waves)
        elif worst > size:
            needed = _format.buffer_bytes_for_partition_slots(worst)
            raise RuntimeError(
                f"trace overflowed: waves claimed {worst} slots in one of the {len(claims)} partitions, each "
                f"holds {size}. Re-run with FLYDSL_KTRACE_BUFFER_BYTES>={needed}, "
                f"a block filter (capture(blocks=...) or FLYDSL_KTRACE_BLOCKS), or fewer launches between reads."
            )
        else:
            # Partitions interleave by wave region (region j * partitions + p is the j-th claim of
            # partition p), so the claimed regions form one prefix; unclaimed regions in it are zero.
            nbytes = -(-worst // _format.EVENTS_PER_WAVE) * len(claims) * _format.EVENTS_PER_WAVE * _format.RECORD_BYTES
        if nbytes == 0:
            return []
        first = _format.first_record_slot(len(claims))
        raw = (ctypes.c_char * nbytes)()
        _check(
            _runtime().mgpuTraceBufferRead(
                raw, ctypes.c_void_p(self.device_ptr + first * _format.RECORD_BYTES), nbytes
            ),
            "reading the trace records",
        )
        records, overflowed = decode_records(bytes(raw), first_slot=first)
        if overflowed:
            worst = max(count for _, count in overflowed)
            raise RuntimeError(
                f"trace truncated: {len(overflowed)} wave(s) emitted more than "
                f"{_format.EVENTS_PER_WAVE - 1} events (worst: {worst}). The records past "
                "that point were dropped, so ranges are unbalanced and phases would be "
                "mispaired -- a plausible-looking timeline for the wrong phase. Annotate "
                "at a coarser granularity or sample fewer loop iterations. Increasing "
                "FLYDSL_KTRACE_BUFFER_BYTES or filtering other blocks does not increase "
                "the per-wave event limit."
            )
        return records


def _decode_payload(bits: int, packed: int):
    if not packed & (1 << 31):
        return None
    encoding = (packed >> 27) & 0x7
    if encoding == 0:
        return bits
    if encoding == 1:
        return bits - (1 << 64) if bits & (1 << 63) else bits
    if encoding == 2:
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
    if encoding == 3:
        return struct.unpack("<d", struct.pack("<Q", bits))[0]
    if encoding == 4:
        return bool(bits)
    raise ValueError(f"ktrace: unknown payload type {encoding}")


def decode_records(raw: bytes, first_slot: int = 1) -> tuple[list[dict], list[tuple[int, int]]]:
    """Decode whole wave regions; ``raw`` starts at absolute slot ``first_slot`` with a wave header."""
    per_wave = _format.EVENTS_PER_WAVE
    out = []
    overflowed = []
    dispatch_id = 0
    queue_id = 0
    block = (0, 0, 0)
    wave_id = 0
    for off in range(0, len(raw) - _format.RECORD_BYTES + 1, _format.RECORD_BYTES):
        abs_slot = off // _format.RECORD_BYTES + first_slot
        if per_wave > 0 and (abs_slot - first_slot) % per_wave == 0:
            attempted, wave_word, dispatch_id, queue_id, block_x, block_y = struct.unpack_from("<IIQQII", raw, off)
            block = (block_x, block_y, wave_word >> 8)
            wave_id = wave_word & 0xFF
            if attempted > per_wave - 1:
                overflowed.append((abs_slot // per_wave, attempted))
            continue
        ts, payload, hw_id, xcc_id, packed, start_slot = struct.unpack_from("<QQIIII", raw, off)
        if ts == 0:
            continue
        out.append(
            {
                "ts": ts,
                "payload": _decode_payload(payload, packed),
                "payload_type": _format.PAYLOAD_TYPES[(packed >> 27) & 0x7] if packed & (1 << 31) else None,
                "hw_id": hw_id,
                "xcc_id": xcc_id,
                "dispatch_id": dispatch_id,
                "queue_id": queue_id,
                "block": block,
                "wave_id": wave_id,
                "event_id": packed & 0xFFFFFF,
                "kind": (packed >> 24) & 0x7,
                "start_slot": start_slot,
                "slot": abs_slot,
                "wave": hw_id & 0xF,
                "simd": (hw_id >> 4) & 0x3,
                "cu": (hw_id >> 8) & 0xF,
                "sh": (hw_id >> 12) & 0x1,
                "se": (hw_id >> 13) & 0x3,
            }
        )
    return out, overflowed


def buffer_bytes() -> int:
    from ...utils import env

    return int(env.ktrace.buffer_bytes)


def allocate(capacity_bytes: int | None = None) -> TraceBuffer:
    capacity_bytes = capacity_bytes if capacity_bytes is not None else buffer_bytes()
    capacity_slots = capacity_bytes // _format.RECORD_BYTES
    if capacity_slots == 0:
        raise ValueError(f"FLYDSL_KTRACE_BUFFER_BYTES={capacity_bytes} is smaller than one record")

    rt = _runtime()
    address = rt.mgpuMemAlloc(capacity_bytes, None, False)
    if not address:
        raise RuntimeError(f"ktrace: could not allocate a {capacity_bytes}-byte trace buffer")
    device_ptr = ctypes.c_void_p(address)
    _check(rt.mgpuTraceBufferClear(device_ptr, capacity_bytes), "clearing the trace buffer")

    return TraceBuffer(
        device_ptr=device_ptr.value or 0,
        capacity_slots=capacity_slots - 1,
        device_id=_current_device(),
    )


def dump_dir() -> str:
    from ...utils import env

    return env.ktrace.dump_dir or os.getcwd()
