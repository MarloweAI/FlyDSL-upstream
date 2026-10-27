# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


RECORD_BYTES = 32
PAYLOAD_TYPES = ("uint64", "int64", "float32", "float64", "bool")

EVENTS_PER_WAVE = 256

UNPAIRED_SLOT = 0xFFFFFFFF

# Waves claim their slots from one of up to MAX_PARTITIONS 64-bit cursors, one per
# CURSOR_STRIDE_BYTES line, starting at slot 1. Must match kCursorStrideBytes in the pass.
CURSOR_STRIDE_BYTES = 128
MAX_PARTITIONS = 256


def first_record_slot(partitions: int) -> int:
    return 1 + partitions * (CURSOR_STRIDE_BYTES // RECORD_BYTES)


def partition_layout(capacity_slots: int) -> tuple[int, int]:
    """Return ``(partitions, slots per partition)`` for a buffer of ``capacity_slots`` record slots.

    Uses the most partitions, up to MAX_PARTITIONS, that still give each partition room for one wave.
    """
    partitions = MAX_PARTITIONS
    while partitions > 1 and _slots_per_partition(capacity_slots, partitions) < EVENTS_PER_WAVE:
        partitions //= 2
    return partitions, _slots_per_partition(capacity_slots, partitions)


def buffer_bytes_for_partition_slots(slots: int) -> int:
    """Smallest buffer size whose partitions each hold at least ``slots`` record slots."""
    per_partition = -(-max(slots, EVENTS_PER_WAVE) // EVENTS_PER_WAVE) * EVENTS_PER_WAVE
    capacity_slots = first_record_slot(MAX_PARTITIONS) - 1 + MAX_PARTITIONS * per_partition
    return (capacity_slots + 1) * RECORD_BYTES


def _slots_per_partition(capacity_slots: int, partitions: int) -> int:
    usable = capacity_slots + 1 - first_record_slot(partitions)
    return max(usable, 0) // partitions // EVENTS_PER_WAVE * EVENTS_PER_WAVE
