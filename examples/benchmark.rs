// Copyright 2026 Raul Mc
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reproducible GPU benchmark harness for myelin-accelerator kernels.
//!
//! # Usage
//!
//! ```bash
//! # CPU-only bitpacking benchmarks
//! cargo run --example benchmark --features bench
//!
//! # GPU kernel benchmarks
//! cargo run --example benchmark --features bench,cuda
//!
//! # Compare against a baseline
//! cargo run --example benchmark --features bench -- --baseline results.json
//!
//! # Custom iteration counts
//! cargo run --example benchmark --features bench -- --warmup 20 --iterations 200
//! ```
//!
//! # Output
//!
//! Emits `benchmark_results.json` and `benchmark_results.csv` in the current
//! directory. Results include latency percentiles (p50, p95, p99), throughput,
//! and GPU info when available.
//!
//! # Nsight Profiling
//!
//! For detailed kernel profiling with Nsight Compute:
//!
//! ```bash
//! ncu --set full -o profile cargo run --example benchmark --features bench,cuda
//! ```
//!
//! For timeline profiling with Nsight Systems:
//!
//! ```bash
//! nsys profile -o timeline cargo run --example benchmark --features bench,cuda
//! ```

use std::time::{Duration, Instant};

// ── CLI argument parsing (minimal, no clap dependency) ──────────────────────

struct Config {
    warmup: usize,
    iterations: usize,
    baseline: Option<String>,
    output_prefix: String,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut warmup = 10;
        let mut iterations = 100;
        let mut baseline = None;
        let mut output_prefix = "benchmark_results".to_string();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--warmup" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("--warmup requires a value");
                        std::process::exit(1);
                    }
                    warmup = args[i].parse().expect("--warmup requires a number");
                }
                "--iterations" | "-n" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("--iterations requires a value");
                        std::process::exit(1);
                    }
                    iterations = args[i].parse().expect("--iterations requires a number");
                }
                "--baseline" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("--baseline requires a value");
                        std::process::exit(1);
                    }
                    baseline = Some(args[i].clone());
                }
                "--output" | "-o" => {
                    i += 1;
                    if i >= args.len() {
                        eprintln!("--output requires a value");
                        std::process::exit(1);
                    }
                    output_prefix = args[i].clone();
                }
                "--help" | "-h" => {
                    println!("Usage: benchmark [OPTIONS]");
                    println!();
                    println!("Options:");
                    println!("  --warmup <N>        Warmup iterations (default: 10)");
                    println!("  --iterations <N>    Timed iterations (default: 100)");
                    println!("  --baseline <FILE>   Compare against a previous JSON result");
                    println!(
                        "  --output <PREFIX>   Output file prefix (default: benchmark_results)"
                    );
                    println!("  -h, --help          Show this help");
                    std::process::exit(0);
                }
                other => {
                    eprintln!("Unknown argument: {other}");
                    std::process::exit(1);
                }
            }
            i += 1;
        }

        if iterations == 0 {
            eprintln!("--iterations must be > 0");
            std::process::exit(1);
        }

        Config {
            warmup,
            iterations,
            baseline,
            output_prefix,
        }
    }
}

// ── Benchmark result types ──────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct BenchmarkResult {
    name: String,
    iterations: usize,
    total_duration_us: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    min_us: f64,
    max_us: f64,
    throughput_ops_per_sec: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct BenchmarkReport {
    timestamp: String,
    gpu_info: Option<GpuInfo>,
    config: RunConfig,
    results: Vec<BenchmarkResult>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct GpuInfo {
    device_name: String,
    driver_version: String,
    cuda_version: String,
    sm_arch: String,
    vram_total_mb: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct RunConfig {
    warmup: usize,
    iterations: usize,
}

// ── Benchmark runner ────────────────────────────────────────────────────────

fn run_benchmark<F: FnMut()>(
    name: &str,
    warmup: usize,
    iterations: usize,
    mut f: F,
) -> BenchmarkResult {
    // Warmup
    for _ in 0..warmup {
        f();
    }

    // Timed iterations
    let mut durations = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        durations.push(start.elapsed());
    }

    durations.sort();
    let total: Duration = durations.iter().sum();
    let total_us = total.as_secs_f64() * 1e6;
    let mean_us = total_us / iterations as f64;

    let percentile = |p: f64| -> f64 {
        // Nearest-rank method: rank = ceil(p/100 * N), then convert to 0-based index.
        let rank = ((p / 100.0) * iterations as f64).ceil() as usize;
        let idx = rank.saturating_sub(1).min(iterations - 1);
        durations[idx].as_secs_f64() * 1e6
    };

    let min_us = durations[0].as_secs_f64() * 1e6;
    let max_us = durations[iterations - 1].as_secs_f64() * 1e6;
    let p50_us = percentile(50.0);
    let p95_us = percentile(95.0);
    let p99_us = percentile(99.0);
    let throughput = if mean_us > 0.0 {
        1_000_000.0 / mean_us
    } else {
        f64::INFINITY
    };

    BenchmarkResult {
        name: name.to_string(),
        iterations,
        total_duration_us: total_us,
        mean_us,
        p50_us,
        p95_us,
        p99_us,
        min_us,
        max_us,
        throughput_ops_per_sec: throughput,
    }
}

// ── GPU info collection ─────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn collect_gpu_info() -> Option<GpuInfo> {
    // When CUDA is available, collect device info via cust.
    // This is best-effort; return None if anything fails.
    use myelin_accelerator::GpuContext;
    if GpuContext::init().is_err() {
        return None;
    }
    // TODO: Use cust::device::Device to query actual hardware properties.
    // Return None until real device queries are implemented — avoids
    // emitting misleading "unknown" fields in benchmark reports.
    None
}

#[cfg(not(feature = "cuda"))]
fn collect_gpu_info() -> Option<GpuInfo> {
    None
}

// ── Bitpacking benchmarks ───────────────────────────────────────────────────

fn bench_bitpacking(config: &Config) -> Vec<BenchmarkResult> {
    use myelin_accelerator::bitpacking::{
        pack_binary, pack_ternary, unpack_binary, unpack_ternary,
    };

    let mut results = Vec::new();

    // Binary packing - small (256 elements)
    let binary_small: Vec<bool> = (0..256).map(|i| i % 3 == 0).collect();
    results.push(run_benchmark(
        "bitpack_binary_pack_256",
        config.warmup,
        config.iterations,
        || {
            let _ = pack_binary(&binary_small);
        },
    ));

    // Binary unpacking - small
    let packed_small = pack_binary(&binary_small);
    results.push(run_benchmark(
        "bitpack_binary_unpack_256",
        config.warmup,
        config.iterations,
        || {
            let _ = unpack_binary(&packed_small, Some(256));
        },
    ));

    // Binary packing - large (65536 elements)
    let binary_large: Vec<bool> = (0..65536).map(|i| i % 5 < 2).collect();
    results.push(run_benchmark(
        "bitpack_binary_pack_65536",
        config.warmup,
        config.iterations,
        || {
            let _ = pack_binary(&binary_large);
        },
    ));

    // Binary unpacking - large
    let packed_large = pack_binary(&binary_large);
    results.push(run_benchmark(
        "bitpack_binary_unpack_65536",
        config.warmup,
        config.iterations,
        || {
            let _ = unpack_binary(&packed_large, Some(65536));
        },
    ));

    // Ternary packing - small (256 elements)
    let ternary_small: Vec<i8> = (0..256)
        .map(|i| match i % 3 {
            0 => 0i8,
            1 => 1i8,
            _ => -1i8,
        })
        .collect();
    results.push(run_benchmark(
        "bitpack_ternary_pack_256",
        config.warmup,
        config.iterations,
        || {
            let _ = pack_ternary(&ternary_small);
        },
    ));

    // Ternary unpacking - small
    let packed_tsmall = pack_ternary(&ternary_small);
    results.push(run_benchmark(
        "bitpack_ternary_unpack_256",
        config.warmup,
        config.iterations,
        || {
            let _ = unpack_ternary(&packed_tsmall, Some(256));
        },
    ));

    // Ternary packing - large (65536 elements)
    let ternary_large: Vec<i8> = (0..65536)
        .map(|i| match i % 3 {
            0 => 0i8,
            1 => 1i8,
            _ => -1i8,
        })
        .collect();
    results.push(run_benchmark(
        "bitpack_ternary_pack_65536",
        config.warmup,
        config.iterations,
        || {
            let _ = pack_ternary(&ternary_large);
        },
    ));

    // Ternary unpacking - large
    let packed_tlarge = pack_ternary(&ternary_large);
    results.push(run_benchmark(
        "bitpack_ternary_unpack_65536",
        config.warmup,
        config.iterations,
        || {
            let _ = unpack_ternary(&packed_tlarge, Some(65536));
        },
    ));

    results
}

// ── GPU kernel benchmarks (requires cuda feature) ───────────────────────────

#[cfg(feature = "cuda")]
fn bench_gpu_kernels(config: &Config) -> Vec<BenchmarkResult> {
    use myelin_accelerator::{GpuAccelerator, GpuBuffer};

    let mut results = Vec::new();
    let acc = GpuAccelerator::new();
    if !acc.is_ready() {
        eprintln!("[bench] GPU not available, skipping kernel benchmarks");
        return results;
    }

    // Poisson encoding benchmark
    let n = 4096;
    let stimuli = GpuBuffer::from_slice(&vec![0.5f32; n]).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(n).unwrap();
    results.push(run_benchmark(
        "poisson_encode_4096",
        config.warmup,
        config.iterations,
        || {
            acc.poisson_encode(&stimuli, &mut spikes, 42).unwrap();
        },
    ));

    // Satsolver extract benchmark
    let n_vars = 1024;
    let n_walkers = 256;
    let assignment = GpuBuffer::from_slice(&vec![0u8; n_vars * n_walkers]).unwrap();
    let best_walker = GpuBuffer::from_slice(&[0i32]).unwrap();
    let mut output = GpuBuffer::<u8>::alloc(n_vars).unwrap();
    results.push(run_benchmark(
        "satsolver_extract_1024x256",
        config.warmup,
        config.iterations,
        || {
            acc.satsolver_extract(
                &assignment,
                &best_walker,
                &mut output,
                n_vars as i32,
                n_walkers as i32,
            )
            .unwrap();
        },
    ));

    results
}

#[cfg(not(feature = "cuda"))]
fn bench_gpu_kernels(_config: &Config) -> Vec<BenchmarkResult> {
    eprintln!("[bench] Built without cuda feature, skipping GPU kernel benchmarks");
    Vec::new()
}

// ── Baseline comparison ─────────────────────────────────────────────────────

fn compare_with_baseline(current: &[BenchmarkResult], baseline_path: &str) {
    let Ok(data) = std::fs::read_to_string(baseline_path) else {
        eprintln!("[bench] Could not read baseline file: {baseline_path}");
        return;
    };
    let Ok(report) = serde_json::from_str::<BenchmarkReport>(&data) else {
        eprintln!("[bench] Could not parse baseline JSON");
        return;
    };

    println!("\n{:=>70}", "");
    println!("  Baseline comparison: {baseline_path}");
    println!("{:=>70}\n", "");

    println!(
        "{:<40} {:>12} {:>12} {:>10}",
        "Benchmark", "Baseline(µs)", "Current(µs)", "Change"
    );
    println!("{:-<76}", "");

    for curr in current {
        if let Some(base) = report.results.iter().find(|r| r.name == curr.name) {
            let change = if base.mean_us > 0.0 {
                ((curr.mean_us - base.mean_us) / base.mean_us) * 100.0
            } else {
                0.0
            };
            let indicator = if change > 5.0 {
                "⚠ SLOWER"
            } else if change < -5.0 {
                "✓ faster"
            } else {
                "  ~same"
            };
            println!(
                "{:<40} {:>12.2} {:>12.2} {:>+8.1}% {}",
                curr.name, base.mean_us, curr.mean_us, change, indicator
            );
        }
    }
}

// ── Output writers ──────────────────────────────────────────────────────────

fn write_json(report: &BenchmarkReport, prefix: &str) {
    let path = format!("{prefix}.json");
    let data = serde_json::to_string_pretty(report).expect("serialize report");
    std::fs::write(&path, data).expect("write JSON");
    println!("[bench] Results written to {path}");
}

fn write_csv(results: &[BenchmarkResult], prefix: &str) {
    let path = format!("{prefix}.csv");
    let mut csv = String::from(
        "name,iterations,mean_us,p50_us,p95_us,p99_us,min_us,max_us,throughput_ops_sec\n",
    );
    for r in results {
        csv.push_str(&format!(
            "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.0}\n",
            r.name,
            r.iterations,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.p99_us,
            r.min_us,
            r.max_us,
            r.throughput_ops_per_sec,
        ));
    }
    std::fs::write(&path, csv).expect("write CSV");
    println!("[bench] Results written to {path}");
}

// ── Pretty printer ──────────────────────────────────────────────────────────

fn print_results(results: &[BenchmarkResult]) {
    println!("\n{:=>90}", "");
    println!("  myelin-accelerator benchmark results");
    println!("{:=>90}\n", "");

    println!(
        "{:<40} {:>8} {:>10} {:>10} {:>10} {:>12}",
        "Benchmark", "Iters", "Mean(µs)", "P50(µs)", "P95(µs)", "Ops/sec"
    );
    println!("{:-<92}", "");

    for r in results {
        println!(
            "{:<40} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>12.0}",
            r.name, r.iterations, r.mean_us, r.p50_us, r.p95_us, r.throughput_ops_per_sec,
        );
    }
    println!();
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let config = Config::from_args();

    println!(
        "[bench] Warmup: {}, Iterations: {}",
        config.warmup, config.iterations
    );

    let gpu_info = collect_gpu_info();
    if let Some(ref info) = gpu_info {
        println!("[bench] GPU: {} ({})", info.device_name, info.sm_arch);
    } else {
        println!("[bench] No GPU detected (CPU-only benchmarks)");
    }

    let mut results = Vec::new();
    results.extend(bench_bitpacking(&config));
    results.extend(bench_gpu_kernels(&config));

    print_results(&results);

    let report = BenchmarkReport {
        timestamp: chrono_now(),
        gpu_info,
        config: RunConfig {
            warmup: config.warmup,
            iterations: config.iterations,
        },
        results: results.clone(),
    };

    write_json(&report, &config.output_prefix);
    write_csv(&results, &config.output_prefix);

    if let Some(ref baseline) = config.baseline {
        compare_with_baseline(&results, baseline);
    }

    println!("[bench] Done.");
}

/// Simple timestamp without chrono dependency.
fn chrono_now() -> String {
    // Use system time to produce an ISO-8601-ish timestamp.
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Simple conversion: days since epoch -> Y-M-D (ignores leap seconds)
    let days = secs / 86400;
    let (y, m, d) = days_to_ymd(days as i64 + 719468);
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

fn days_to_ymd(g: i64) -> (i64, u32, u32) {
    let mut y = (10000 * g + 14780) / 3652425;
    let mut doy = g - (365 * y + y / 4 - y / 100 + y / 400);
    if doy < 0 {
        y -= 1;
        doy = g - (365 * y + y / 4 - y / 100 + y / 400);
    }
    let mi = (100 * doy + 52) / 3060;
    let month = (mi + 2) % 12 + 1;
    let year = y + (mi + 2) / 12;
    let day = doy - (mi * 306 + 5) / 10 + 1;
    (year, month as u32, day as u32)
}
