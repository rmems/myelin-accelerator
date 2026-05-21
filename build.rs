use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const PTX_STUB_VERSION: &str = "8.5";
const PTX_STUB: &str = ".version 8.5\n.target sm_80\n.address_size 64\n";

const KERNELS: &[(&str, &str)] = &[
    ("spiking_network.cu", "spiking_network_sm_120.ptx"),
    ("vector_similarity.cu", "vector_similarity_sm_120.ptx"),
    ("satsolver.cu", "satsolver_sm_120.ptx"),
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let cu_dir = manifest_dir.join("cu");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC");
    println!("cargo:rerun-if-env-changed=MYELIN_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=MYELIN_PTX_VERSION");
    println!("cargo:rerun-if-env-changed=MYELIN_NVCC_THREADS");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cu/common.cuh");
    for &(cu_name, _) in KERNELS {
        println!("cargo:rerun-if-changed=cu/{cu_name}");
    }

    let nvcc = find_nvcc();
    let arch = env::var("MYELIN_CUDA_ARCH").unwrap_or_else(|_| "sm_120".to_string());
    let ptx_version =
        env::var("MYELIN_PTX_VERSION").unwrap_or_else(|_| PTX_STUB_VERSION.to_string());

    let Some(nvcc_path) = nvcc else {
        emit_stub_ptx(&out_dir);
        println!(
            "cargo:warning=nvcc not found; wrote stub PTX files. Enable CUDA by installing toolkit or setting CUDA_NVCC."
        );
        return;
    };

    match nvcc_version(&nvcc_path) {
        Some(v) => println!("cargo:warning=using nvcc: {v}"),
        None => println!("cargo:warning=using nvcc at {}", nvcc_path.display()),
    }

    for &(cu_name, ptx_name) in KERNELS {
        let source = cu_dir.join(cu_name);
        let output = out_dir.join(ptx_name);
        compile_to_ptx(&nvcc_path, &cu_dir, &source, &output, &arch);
        patch_ptx_version_any(&output, &ptx_version);
        println!("cargo:warning=compiled {cu_name} -> {ptx_name} (arch={arch}, ptx={ptx_version})");
    }
}

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(path) = env::var("CUDA_NVCC") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    for root_var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = env::var(root_var) {
            let p = PathBuf::from(root).join("bin").join(exe_name("nvcc"));
            if p.exists() {
                return Some(p);
            }
        }
    }

    let candidate = PathBuf::from(exe_name("nvcc"));
    Command::new(&candidate)
        .arg("--version")
        .status()
        .ok()
        .filter(|s| s.success())
        .map(|_| candidate)
}

fn exe_name(base: &str) -> OsString {
    if cfg!(windows) {
        format!("{base}.exe").into()
    } else {
        base.into()
    }
}

fn nvcc_version(nvcc: &Path) -> Option<String> {
    let out = Command::new(nvcc).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()
        .and_then(|s| s.lines().last().map(|l| l.trim().to_string()))
}

fn compile_to_ptx(nvcc: &Path, cu_dir: &Path, source: &Path, output: &Path, arch: &str) {
    let threads = env::var("MYELIN_NVCC_THREADS").unwrap_or_else(|_| "0".to_string());

    let status = Command::new(nvcc)
        .arg("-ptx")
        .arg(format!("-arch={arch}"))
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--restrict")
        .arg("--threads")
        .arg(threads)
        .arg("--allow-unsupported-compiler")
        .arg("-I")
        .arg(cu_dir)
        .arg("-o")
        .arg(output)
        .arg(source)
        .status()
        .unwrap_or_else(|e| panic!("Failed to invoke nvcc for {}: {e}", source.display()));

    if !status.success() {
        panic!("nvcc failed to compile {}", source.display());
    }

    assert!(
        output.exists(),
        "nvcc completed but did not emit {}",
        output.display()
    );
}

fn emit_stub_ptx(out_dir: &Path) {
    for &(_, ptx_name) in KERNELS {
        fs::write(out_dir.join(ptx_name), PTX_STUB)
            .unwrap_or_else(|e| panic!("Failed to write stub PTX {ptx_name}: {e}"));
    }
}

fn patch_ptx_version_any(path: &Path, new_ver: &str) {
    let Ok(text) = fs::read_to_string(path) else {
        return;
    };
    let mut replaced = false;
    let patched = text
        .lines()
        .map(|line| {
            if line.trim_start().starts_with(".version ") {
                replaced = true;
                format!(".version {new_ver}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    if replaced {
        fs::write(path, format!("{patched}\n"))
            .unwrap_or_else(|e| panic!("Failed to patch PTX version in {}: {e}", path.display()));
    }
}
