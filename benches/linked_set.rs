use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashSet;
use toblerone::LinkedSet;

fn create_set(size: usize) {
    let mut ls: LinkedSet<usize> = LinkedSet::new();
    for i in 0..size {
        ls.insert(i);
    }
}

fn std_create_set(size: usize) {
    let mut hs: HashSet<usize> = HashSet::new();
    for i in 0..size {
        hs.insert(i);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut c = Criterion::default()
        .warm_up_time(std::time::Duration::new(10, 0))
        .measurement_time(std::time::Duration::new(15, 0))
        .sample_size(100);

    c.bench_function("insert 100k", |b| {
        b.iter(|| black_box(create_set(black_box(100_000))))
    });
    c.bench_function("insert 100k STD", |b| {
        b.iter(|| black_box(std_create_set(black_box(100_000))))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
