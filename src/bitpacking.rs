// Copyright 2026 Raul Mc
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Binary and ternary bitpacking utilities.
//!
//! # Packing Layouts
//!
//! ## Binary (1-bit)
//! Each `u32` word holds 32 binary values. Bit `i` of word `w` encodes
//! element `w * 32 + i`. A set bit means `true`/`1`, a clear bit means
//! `false`/`0`.
//!
//! ## Ternary (2-bit)
//! Each `u32` word holds 16 ternary values. Bits `(2*i, 2*i+1)` of word
//! `w` encode element `w * 16 + i`:
//! - `0b00` → `0`
//! - `0b01` → `+1`
//! - `0b10` → `-1`
//! - `0b11` → `0` (reserved, decoded as zero)
//!
//! # Alignment
//! Packed buffers are naturally 4-byte aligned (they contain `u32`).
//! For CUDA vectorized 128-bit loads, ensure 16-byte alignment.

/// Values packed into a single `u32` for binary encoding.
pub const BINARY_VALUES_PER_WORD: usize = 32;

/// Values packed into a single `u32` for ternary encoding.
pub const TERNARY_VALUES_PER_WORD: usize = 16;

// ── Binary packing ──────────────────────────────────────────────────────────

/// Number of `u32` words needed to pack `n` binary values.
pub fn binary_word_count(n: usize) -> usize {
    n.div_ceil(BINARY_VALUES_PER_WORD)
}

/// Pack boolean values into a dense bit vector.
///
/// `true` → bit set (1), `false` → bit clear (0).
///
/// # Panics
/// Never panics; output length is always `binary_word_count(values.len())`.
pub fn pack_binary(values: &[bool]) -> Vec<u32> {
    let word_count = binary_word_count(values.len());
    let mut out = vec![0u32; word_count];
    for (i, &v) in values.iter().enumerate() {
        if v {
            out[i / BINARY_VALUES_PER_WORD] |= 1u32 << (i % BINARY_VALUES_PER_WORD);
        }
    }
    out
}

/// Unpack a dense bit vector back into boolean values.
///
/// Returns `n` values, where `n = values.len() * 32` if `count` is `None`.
/// If `count` is provided, returns exactly that many values (truncating
/// any bits beyond `count` in the last word).
pub fn unpack_binary(packed: &[u32], count: Option<usize>) -> Vec<bool> {
    let total_bits = count.unwrap_or(packed.len() * BINARY_VALUES_PER_WORD);
    let mut out = Vec::with_capacity(total_bits);
    for i in 0..total_bits {
        let word = i / BINARY_VALUES_PER_WORD;
        let bit = i % BINARY_VALUES_PER_WORD;
        out.push((packed[word] >> bit) & 1 != 0);
    }
    out
}

// ── Ternary packing ─────────────────────────────────────────────────────────

/// Number of `u32` words needed to pack `n` ternary values.
pub fn ternary_word_count(n: usize) -> usize {
    n.div_ceil(TERNARY_VALUES_PER_WORD)
}

/// Pack signed ternary values (`{-1, 0, +1}`) into dense 2-bit encoding.
///
/// Encoding: `0` → `0b00`, `+1` → `0b01`, `-1` → `0b10`.
/// Values outside `{-1, 0, +1}` are clamped.
pub fn pack_ternary(values: &[i8]) -> Vec<u32> {
    let word_count = ternary_word_count(values.len());
    let mut out = vec![0u32; word_count];
    for (i, &v) in values.iter().enumerate() {
        let code: u32 = match v {
            0 => 0b00,
            1..=i8::MAX => 0b01, // +1 and above clamped to +1
            i8::MIN..=-1 => 0b10, // -1 and below clamped to -1
        };
        let word = i / TERNARY_VALUES_PER_WORD;
        let bit = (i % TERNARY_VALUES_PER_WORD) * 2;
        out[word] |= code << bit;
    }
    out
}

/// Unpack a dense 2-bit ternary vector back into signed values.
///
/// Returns `n` values, where `n = values.len() * 16` if `count` is `None`.
/// If `count` is provided, returns exactly that many values.
///
/// Decoding: `0b00` → `0`, `0b01` → `+1`, `0b10` → `-1`, `0b11` → `0`.
pub fn unpack_ternary(packed: &[u32], count: Option<usize>) -> Vec<i8> {
    let total = count.unwrap_or(packed.len() * TERNARY_VALUES_PER_WORD);
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let word = i / TERNARY_VALUES_PER_WORD;
        let bit = (i % TERNARY_VALUES_PER_WORD) * 2;
        let code = (packed[word] >> bit) & 0b11;
        let val = match code {
            0b01 => 1i8,
            0b10 => -1i8,
            _ => 0i8, // 0b00 and 0b11 both decode to 0
        };
        out.push(val);
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Binary roundtrip ────────────────────────────────────────────────────

    #[test]
    fn binary_empty() {
        let packed = pack_binary(&[]);
        assert!(packed.is_empty());
        let unpacked = unpack_binary(&packed, Some(0));
        assert!(unpacked.is_empty());
    }

    #[test]
    fn binary_single_word() {
        let values: Vec<bool> = (0..32).map(|i| i % 3 == 0).collect();
        let packed = pack_binary(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_binary(&packed, Some(32));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn binary_multi_word() {
        let values: Vec<bool> = (0..100).map(|i| i % 7 < 3).collect();
        let packed = pack_binary(&values);
        assert_eq!(packed.len(), binary_word_count(100));
        let unpacked = unpack_binary(&packed, Some(100));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn binary_all_true() {
        let values = vec![true; 64];
        let packed = pack_binary(&values);
        assert_eq!(packed, vec![0xFFFF_FFFF, 0xFFFF_FFFF]);
        let unpacked = unpack_binary(&packed, Some(64));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn binary_all_false() {
        let values = vec![false; 64];
        let packed = pack_binary(&values);
        assert_eq!(packed, vec![0u32, 0u32]);
        let unpacked = unpack_binary(&packed, Some(64));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn binary_exact_boundary() {
        // Exactly 32 values → 1 word
        let values = vec![true; 32];
        let packed = pack_binary(&values);
        assert_eq!(packed.len(), 1);
        // 33 values → 2 words
        let mut values33 = values;
        values33.push(false);
        let packed = pack_binary(&values33);
        assert_eq!(packed.len(), 2);
    }

    // ── Ternary roundtrip ───────────────────────────────────────────────────

    #[test]
    fn ternary_empty() {
        let packed = pack_ternary(&[]);
        assert!(packed.is_empty());
        let unpacked = unpack_ternary(&packed, Some(0));
        assert!(unpacked.is_empty());
    }

    #[test]
    fn ternary_single_word() {
        let values: Vec<i8> = vec![0, 1, -1, 0, 1, 1, -1, -1, 0, 1, -1, 0, 1, -1, 0, 0];
        let packed = pack_ternary(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_ternary(&packed, Some(16));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn ternary_multi_word() {
        let values: Vec<i8> = (0..50)
            .map(|i| match i % 3 {
                0 => 0i8,
                1 => 1i8,
                _ => -1i8,
            })
            .collect();
        let packed = pack_ternary(&values);
        assert_eq!(packed.len(), ternary_word_count(50));
        let unpacked = unpack_ternary(&packed, Some(50));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn ternary_all_zero() {
        let values = vec![0i8; 32];
        let packed = pack_ternary(&values);
        assert_eq!(packed, vec![0u32, 0u32]);
        let unpacked = unpack_ternary(&packed, Some(32));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn ternary_clamping() {
        // Values outside {-1, 0, +1} should be clamped
        let packed = pack_ternary(&[100i8, -100i8, 0i8]);
        let unpacked = unpack_ternary(&packed, Some(3));
        assert_eq!(unpacked, vec![1i8, -1i8, 0i8]);
    }

    #[test]
    fn ternary_reserved_decodes_to_zero() {
        // Manually craft a word with 0b11 in one slot
        let mut word = 0u32;
        // Put 0b11 in slot 0 (bits 0-1)
        word |= 0b11;
        // Put 0b01 in slot 1 (bits 2-3)
        word |= 0b01 << 2;
        let unpacked = unpack_ternary(&[word], Some(2));
        assert_eq!(unpacked[0], 0i8); // 0b11 → 0
        assert_eq!(unpacked[1], 1i8); // 0b01 → +1
    }

    // ── Word count helpers ──────────────────────────────────────────────────

    #[test]
    fn word_count_helpers() {
        assert_eq!(binary_word_count(0), 0);
        assert_eq!(binary_word_count(1), 1);
        assert_eq!(binary_word_count(32), 1);
        assert_eq!(binary_word_count(33), 2);

        assert_eq!(ternary_word_count(0), 0);
        assert_eq!(ternary_word_count(1), 1);
        assert_eq!(ternary_word_count(16), 1);
        assert_eq!(ternary_word_count(17), 2);
    }

    // ── Property-based roundtrip tests ──────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn binary_roundtrip(values in proptest::collection::vec(any::<bool>(), 0..=512)) {
            let packed = pack_binary(&values);
            let unpacked = unpack_binary(&packed, Some(values.len()));
            prop_assert_eq!(&values, &unpacked);
        }

        #[test]
        fn binary_word_count_never_zero_for_nonempty(
            n in 1usize..=10_000
        ) {
            prop_assert!(binary_word_count(n) > 0);
        }

        #[test]
        fn ternary_roundtrip(
            values in proptest::collection::vec(
                prop_oneof![Just(0i8), Just(1i8), Just(-1i8)],
                0..=512
            )
        ) {
            let packed = pack_ternary(&values);
            let unpacked = unpack_ternary(&packed, Some(values.len()));
            prop_assert_eq!(&values, &unpacked);
        }

        #[test]
        fn ternary_clamp_roundtrip(
            values in proptest::collection::vec(any::<i8>(), 0..=512)
        ) {
            // Pack raw values (they get clamped), then unpack and verify
            // each element matches the clamped input.
            let packed = pack_ternary(&values);
            let unpacked = unpack_ternary(&packed, Some(values.len()));
            for (i, (&inp, &out)) in values.iter().zip(unpacked.iter()).enumerate() {
                let expected = match inp {
                    0 => 0i8,
                    1..=i8::MAX => 1i8,
                    i8::MIN..=-1 => -1i8,
                };
                prop_assert_eq!(out, expected, "mismatch at index {}: input={}", i, inp);
            }
        }

        #[test]
        fn ternary_word_count_never_zero_for_nonempty(
            n in 1usize..=10_000
        ) {
            prop_assert!(ternary_word_count(n) > 0);
        }
    }
}
