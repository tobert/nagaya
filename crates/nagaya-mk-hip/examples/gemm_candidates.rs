//! Print the hipBLASLt candidates of the GEMM shapes of Qwen-Image, with their kernel names and workspaces.
//! Stop `nagaya.service` first (spec 3.1). Run: `cargo run -p nagaya-mk-hip --features hip --example gemm_candidates`.

use nagaya_mk_hip::gemm::{Blaslt, lineage};
use nagaya_mk_hip::plan::{DataType, GemmShape};

fn main() {
    println!("lineage: {}", lineage().unwrap());
    let blaslt = Blaslt::new().unwrap();
    // (rows, inputs, outputs): the encoder's q, kv and MLP; the transformer's qkv, MLP and out at 1024 by 1024.
    let shapes = [
        (64, 4096, 4096),
        (64, 4096, 1024),
        (64, 4096, 12288),
        (64, 12288, 4096),
        (5106, 4096, 4096),
        (4096, 4096, 16384),
        (4096, 16384, 4096),
        (4096, 4096, 64),
    ];
    for (rows, inputs, outputs) in shapes {
        let shape = GemmShape::linear(rows, inputs, outputs, DataType::Bf16);
        println!("== rows {rows} in {inputs} out {outputs}");
        for (index, workspace) in blaslt.candidates(&shape, 64 << 20, 16).unwrap() {
            println!("  {index:>6} ws {workspace:>9} {}", blaslt.kernel_name(index).unwrap());
        }
    }
}
