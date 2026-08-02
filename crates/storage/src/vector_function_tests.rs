// This module is included from `lib.rs` to test registered SQLite UDFs.

use rusqlite::{Connection, params};

use super::register_vector_functions;

#[test]
fn normalized_vector_dot_matches_cosine_similarity() {
    let connection = Connection::open_in_memory().expect("connection");
    register_vector_functions(&connection).expect("functions");
    let left = [0.6_f32, 0.8_f32];
    let right = [0.8_f32, 0.6_f32];
    let left = left
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let right = right
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let (cosine, dot): (f64, f64) = connection
        .query_row(
            "SELECT cosine_similarity(?1, ?2, 2), vector_dot(?1, ?2, 2)",
            params![left, right],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("score");

    assert!((cosine - dot).abs() < 1e-6);
}
