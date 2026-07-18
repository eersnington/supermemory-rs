use std::path::PathBuf;

use memory_engine::{BGE_DIMENSIONS, EmbeddingModel};

fn supermemory_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".supermemory"))
}

#[test]
fn local_bge_matches_the_v005_embedding_worker() {
    let Some(home) = supermemory_home() else {
        return;
    };
    let model = home.join("models/Xenova/bge-base-en-v1.5");
    let runtime = home.join(
        "runtime/ort-native/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.23.2.dylib",
    );
    if !model.exists() || !runtime.exists() {
        return;
    }

    let model = EmbeddingModel::load(&model, &runtime).expect("local BGE assets should load");
    let vectors = model
        .embed(&["The quick brown fox jumps over the lazy dog.".to_owned()])
        .expect("reference sentence should embed");
    assert_eq!(vectors[0].as_slice().len(), BGE_DIMENSIONS);
    let expected = [
        -0.009_626_173,
        -0.066_305_26,
        0.067_295_52,
        0.031_443_704,
        0.051_232_643,
        -0.004_884_148,
        0.038_636_673,
        0.040_168_07,
    ];
    for (actual, expected) in vectors[0].as_slice().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn normalized_embeddings_support_cosine_similarity() {
    let values = vec![1.0 / 768.0_f32.sqrt(); BGE_DIMENSIONS];
    let vector = memory_engine::EmbeddingVector::new(values).expect("unit vector is valid");
    assert!((vector.similarity(&vector) - 1.0).abs() < 1e-5);
}
