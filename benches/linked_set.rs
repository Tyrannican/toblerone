use criterion::{criterion_group, criterion_main, Criterion};
use std::{collections::HashSet, hint::black_box};
use toblerone::LinkedSet;

fn create_set() {
    let mut ls: LinkedSet<i32> = LinkedSet::new();
    for i in 0..100_000 {
        ls.insert(i);
    }
}

fn std_create_set() {
    let mut hs: HashSet<i32> = HashSet::new();
    for i in 0..100_000 {
        hs.insert(i);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("create set 100k", |b| b.iter(|| black_box(create_set())));
    c.bench_function("create set 100k STD", |b| {
        b.iter(|| black_box(std_create_set()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
