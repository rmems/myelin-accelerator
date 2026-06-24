// Copyright 2026 Raul Mc
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests validating the public API contract.
//!
//! These tests run against the CPU stub (no `cuda` feature) and verify
//! that the public types, methods, and error behaviour are consistent.

use myelin_accelerator::{GpuAccelerator, GpuBuffer, GpuContext, GpuError, KernelModule};

// ── Public type availability ────────────────────────────────────────────────

#[test]
fn public_types_are_exported() {
    // Verify the main types are accessible at the crate root.
    let _: GpuContext;
    let _: KernelModule;
    let _: GpuAccelerator;
}

// ── GpuContext contract ─────────────────────────────────────────────────────

#[test]
fn context_init_returns_error_without_gpu() {
    let result = GpuContext::init();
    assert!(result.is_err());
}

#[test]
fn context_is_available_false_without_gpu() {
    assert!(!GpuContext::is_available());
}

// ── KernelModule contract ───────────────────────────────────────────────────

#[test]
fn kernel_load_returns_error_without_gpu() {
    let result = KernelModule::load();
    assert!(result.is_err());
}

// ── GpuBuffer contract ──────────────────────────────────────────────────────

#[test]
fn buffer_alloc_and_roundtrip() {
    let data = vec![42u32; 128];
    let buf = GpuBuffer::from_slice(&data).expect("from_slice should succeed in stub");
    assert_eq!(buf.len(), 128);
    let out = buf.to_vec().expect("to_vec should succeed in stub");
    assert_eq!(out, data);
}

#[test]
fn buffer_upload_matches_length() {
    let mut buf = GpuBuffer::<i32>::alloc(8).expect("alloc should succeed in stub");
    let data = vec![1i32, 2, 3, 4, 5, 6, 7, 8];
    buf.upload(&data).expect("upload should succeed");
    assert_eq!(buf.to_vec().unwrap(), data);
}

#[test]
fn buffer_upload_rejects_mismatch() {
    let mut buf = GpuBuffer::<i32>::alloc(8).expect("alloc should succeed in stub");
    let result = buf.upload(&[1, 2, 3]);
    assert!(result.is_err());
}

// ── GpuAccelerator contract ─────────────────────────────────────────────────

#[test]
fn accelerator_construction() {
    let acc = GpuAccelerator::new();
    assert!(!acc.is_ready());
    assert!(acc.kernels().is_err());
    assert!(acc.synchronize().is_err());
}

#[test]
fn accelerator_all_kernel_launches_fail_gracefully() {
    let acc = GpuAccelerator::new();

    // satsolver_extract
    let a = GpuBuffer::<u8>::alloc(10).unwrap();
    let b = GpuBuffer::<i32>::alloc(1).unwrap();
    let mut c = GpuBuffer::<u8>::alloc(10).unwrap();
    assert!(acc.satsolver_extract(&a, &b, &mut c, 10, 1).is_err());

    // poisson_encode
    let stim = GpuBuffer::<f32>::alloc(10).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(10).unwrap();
    assert!(acc.poisson_encode(&stim, &mut spikes, 42).is_err());
}

// ── GpuError contract ───────────────────────────────────────────────────────

#[test]
fn error_display_contains_variant_info() {
    let variants = [
        GpuError::NoGpu,
        GpuError::InitFailed("test".into()),
        GpuError::ModuleLoadFailed("test".into()),
        GpuError::KernelNotFound("test".into()),
        GpuError::MemoryError("test".into()),
        GpuError::LaunchFailed("test".into()),
        GpuError::CudaError("test".into()),
    ];

    for err in &variants {
        let msg = err.to_string();
        assert!(!msg.is_empty(), "error Display should not be empty");
        // Each variant should include some distinguishing text
        assert!(msg.len() > 5, "error Display too short: {msg}");
    }
}

#[test]
fn error_implements_std_error() {
    fn check<T: std::error::Error>() {}
    check::<GpuError>();
}

// ── Bitpacking contract ─────────────────────────────────────────────────────

#[test]
fn bitpacking_binary_roundtrip() {
    use myelin_accelerator::bitpacking::{pack_binary, unpack_binary};

    let values: Vec<bool> = (0..100).map(|i| i % 3 == 0).collect();
    let packed = pack_binary(&values);
    let unpacked = unpack_binary(&packed, Some(values.len()));
    assert_eq!(values, unpacked);
}

#[test]
fn bitpacking_ternary_roundtrip() {
    use myelin_accelerator::bitpacking::{pack_ternary, unpack_ternary};

    let values: Vec<i8> = (0..100)
        .map(|i| match i % 3 {
            0 => 0i8,
            1 => 1i8,
            _ => -1i8,
        })
        .collect();
    let packed = pack_ternary(&values);
    let unpacked = unpack_ternary(&packed, Some(values.len()));
    assert_eq!(values, unpacked);
}
