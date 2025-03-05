use std::{
    hash::{DefaultHasher, Hash, Hasher},
    iter::Chain,
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

#[cfg(target_arch = "x86")]
use core::arch::x86 as cpu;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as cpu;

const DEFAULT_CAPACITY: usize = 16;
const BUCKET_SIZE: usize = 16;

const EMPTY: u8 = 0x80;
const TOMB: u8 = 0xFE;

type Link<T> = Option<NonNull<Node<T>>>;

#[derive(Debug)]
struct Node<T> {
    pub value: Rc<T>,
    pub prev: Option<NonNull<Node<T>>>,
    pub next: Option<NonNull<Node<T>>>,
}

impl<T> Node<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: Rc::new(value),
            prev: None,
            next: None,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
struct Bucket<T> {
    meta: Vec<u8>,
    slots: Vec<Option<Rc<T>>>,
}

impl<T> Bucket<T> {
    pub fn new() -> Self {
        Self {
            meta: vec![EMPTY; BUCKET_SIZE],
            slots: vec![None; BUCKET_SIZE],
        }
    }

    pub fn new_bucket_collection(total: usize) -> Vec<Self> {
        (0..total).map(|_| Self::new()).collect()
    }

    unsafe fn simd_lookup(&self, h2: u8) -> u16 {
        let matches = cpu::_mm_set1_epi8(h2 as i8);
        let md_ptr = cpu::_mm_loadu_si128(self.meta.as_ptr() as *const _);
        let cmp = cpu::_mm_cmpeq_epi8(matches, md_ptr);
        cpu::_mm_movemask_epi8(cmp) as u16
    }

    unsafe fn simd_free_or_deleted(&self) -> u16 {
        let md_ptr = cpu::_mm_loadu_si128(self.meta.as_ptr() as *const _);
        let cmp = cpu::_mm_cmplt_epi8(md_ptr, cpu::_mm_setzero_si128());
        cpu::_mm_movemask_epi8(cmp) as u16
    }

    pub fn simd_free_slot(&self) -> Option<usize> {
        let mut idx = 0;
        let mut mask = unsafe { self.simd_free_or_deleted() };

        while mask != 0 {
            if (mask & 1) != 0 {
                return Some(idx);
            }

            mask >>= 1;
            idx += 1;
        }

        None
    }

    pub fn simd_hash_match(&self, h2: u8) -> Vec<usize> {
        let mut idx = 0;
        let mut target_idxs = Vec::new();

        let mut mask = unsafe { self.simd_lookup(h2) };
        while mask != 0 {
            if (mask & 1) != 0 {
                target_idxs.push(idx);
            }

            mask >>= 1;
            idx += 1;
        }

        target_idxs
    }
}

#[derive(Debug)]
pub struct LinkedSet<T> {
    head: Link<T>,
    tail: Link<T>,
    buckets: Vec<Bucket<T>>,
    capacity: usize,
    size: usize,
}

impl<T> LinkedSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    pub fn new() -> Self {
        Self {
            head: None,
            tail: None,
            buckets: Vec::new(),
            capacity: DEFAULT_CAPACITY,
            size: 0,
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            head: None,
            tail: None,
            buckets: Vec::new(),
            capacity: cap,
            size: 0,
        }
    }

    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        if self.should_resize() {
            self.resize();
        }

        if self.contains(&value) {
            return false;
        }

        let (h1, h2) = self.hash(&value);
        if let Some((bucket_idx, slot)) = self.free_slot(h1) {
            let bucket = &mut self.buckets[bucket_idx];
            assert!(bucket.meta[slot] == EMPTY || bucket.meta[slot] == TOMB);
            let node = Box::new(Node::new(value));
            bucket.meta[slot] = h2;
            bucket.slots[slot] = Some(Rc::clone(&node.value));
            self.add_node(node);
            self.size += 1;
            return true;
        } else {
            false
        }
    }

    #[inline]
    pub fn get<'a>(&self, value: &'a T) -> Option<&'a T> {
        let (h1, h2) = self.hash(&value);
        if let Some((bucket_idx, slot)) = self.find(h1, h2) {
            let bucket = &self.buckets[bucket_idx];
            match bucket.slots[slot] {
                Some(ref slot_value) => {
                    if slot_value.as_ref() != value {
                        return None;
                    }

                    return Some(value);
                }
                None => None,
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.get(value).is_some()
    }

    #[inline]
    pub fn remove(&mut self, value: &T) -> bool {
        if self.get(value).is_none() {
            return false;
        }

        let (h1, h2) = self.hash(&value);
        let mut gidx = self.fast_mod(h1 as u32);

        loop {
            if gidx as usize == self.buckets.len() {
                return false;
            }

            let bucket = &mut self.buckets[gidx as usize];
            let potentials = bucket.simd_hash_match(h2);
            let mut found = false;
            'inner: for slot in potentials.iter() {
                match bucket.slots[*slot] {
                    Some(ref slot_value) => {
                        if slot_value.as_ref() != value {
                            gidx += 1;
                            continue;
                        }

                        found = true;
                        bucket.meta[*slot] = TOMB;
                        break 'inner;
                    }
                    None => {}
                }
            }

            if found {
                match self.find_node(value) {
                    Some(node) => {
                        self.remove_node(node);
                        self.size -= 1;
                        return true;
                    }
                    None => unreachable!("if the found flag is set then the node has to exist"),
                }
            }

            gidx += 1;
        }
    }

    #[inline]
    pub fn is_disjoint(&self, other: &LinkedSet<T>) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    #[inline]
    pub fn is_subset(&self, other: &LinkedSet<T>) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| other.contains(v))
        } else {
            false
        }
    }

    #[inline]
    pub fn is_superset(&self, other: &LinkedSet<T>) -> bool {
        other.is_subset(&self)
    }

    #[inline]
    pub fn difference<'a>(&'a self, other: &'a LinkedSet<T>) -> Difference<'a, T> {
        Difference {
            iter: self.iter(),
            other,
        }
    }

    #[inline]
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a LinkedSet<T>,
    ) -> SymmetricDifference<'a, T> {
        SymmetricDifference {
            iter: self.difference(other).chain(other.difference(self)),
        }
    }

    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a LinkedSet<T>) -> Intersection<'a, T> {
        if self.len() <= other.len() {
            Intersection {
                iter: self.iter(),
                other,
            }
        } else {
            Intersection {
                iter: other.iter(),
                other: self,
            }
        }
    }

    #[inline]
    pub fn union<'a>(&'a self, other: &'a LinkedSet<T>) -> Union<'a, T> {
        if self.len() >= other.len() {
            Union {
                iter: self.iter().chain(other.difference(self)),
            }
        } else {
            Union {
                iter: other.iter().chain(self.difference(other)),
            }
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        if self.buckets.is_empty() {
            return;
        }

        // Probably need some custom dropping logic here but meh
        // it's just RCs all the way down
        self.head = None;
        self.tail = None;
        self.buckets = Vec::with_capacity(self.capacity);
        self.size = 0;
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            node: self.head,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn should_resize(&self) -> bool {
        self.buckets.is_empty() || self.size > 4 * self.capacity / 5
    }

    #[inline]
    fn add_node(&mut self, node: Box<Node<T>>) {
        let node = NonNull::new(Box::leak(node));
        if self.head.is_none() {
            self.head = node;
            self.tail = node;
        } else {
            let tail = self.tail.take().expect("if head is set then so is tail");
            unsafe {
                (*tail.as_ptr()).next = node;
                (*node.expect("this is guaranteed to be non-null").as_ptr()).prev = Some(tail);
            }

            self.tail = node;
        }
    }

    #[inline]
    pub(crate) fn insert_rc(&mut self, value: Rc<T>) {
        let (h1, h2) = self.hash(&value);
        let mut gidx = self.fast_mod(h1 as u32);

        loop {
            let bucket = &mut self.buckets[gidx as usize];
            match bucket.simd_free_slot() {
                Some(idx) => {
                    bucket.slots[idx] = Some(value);
                    bucket.meta[idx] = h2;
                    self.size += 1;
                    return;
                }

                None => {
                    gidx += 1;
                }
            }
        }
    }

    #[inline]
    fn remove_node(&mut self, node: NonNull<Node<T>>) {
        // If the value is the head
        let node = unsafe { &*node.as_ptr() };
        if let Some(head) = self.head {
            let h_inner = unsafe { &*head.as_ptr() };
            if h_inner.value == node.value {
                self.head = node.next;
                return;
            }
        }

        // If the node is the tail
        if let Some(tail) = self.tail {
            let t_inner = unsafe { &*tail.as_ptr() };
            if t_inner.value == node.value {
                self.tail = node.prev;
                return;
            }
        }

        // Node is in the middle
        let next = node.next;
        let prev = node.prev;

        assert!(next.is_some());
        assert!(prev.is_some());

        unsafe {
            (*prev.unwrap().as_ptr()).next = next;
            (*next.unwrap().as_ptr()).prev = prev;
        }
    }

    #[inline]
    fn free_slot(&self, h1: u64) -> Option<(usize, usize)> {
        let mut gidx = self.fast_mod(h1 as u32) as usize;

        loop {
            if gidx >= self.buckets.len() {
                return None;
            }

            let bucket = &self.buckets[gidx as usize];
            match bucket.simd_free_slot() {
                Some(idx) => {
                    return Some((gidx, idx));
                }

                None => {
                    gidx += 1;
                }
            }
        }
    }

    #[inline]
    fn find(&self, h1: u64, h2: u8) -> Option<(usize, usize)> {
        let mut gidx = self.fast_mod(h1 as u32);
        loop {
            if gidx as usize >= self.buckets.len() {
                return None;
            }
            let bucket = &self.buckets[gidx as usize];
            let potentials = bucket.simd_hash_match(h2);

            for slot in potentials {
                match bucket.slots[slot] {
                    Some(_) => {
                        if bucket.meta[slot] != h2 {
                            continue;
                        }

                        return Some((gidx as usize, slot));
                    }
                    None => {}
                }
            }

            gidx += 1;
        }
    }

    #[inline]
    fn resize(&mut self) {
        let new_cap = match self.size {
            0 => self.capacity,
            _ => 2 * self.capacity,
        };

        if self.size == 0 {
            self.buckets = Bucket::new_bucket_collection(self.capacity);
            return;
        }

        let mut new_list = Self::with_capacity(new_cap);
        new_list.buckets = Bucket::new_bucket_collection(new_cap);
        for mut bucket in self.buckets.drain(..) {
            for slot in bucket.slots.drain(..) {
                match slot {
                    Some(n) => {
                        new_list.insert_rc(n);
                    }
                    None => {}
                }
            }
        }

        new_list.head = self.head;
        new_list.tail = self.tail;
        *self = new_list;
    }

    #[inline]
    fn find_node(&self, value: &T) -> Option<NonNull<Node<T>>> {
        let mut curr = self.head;
        while let Some(node) = curr {
            let inner = unsafe { &*node.as_ptr() };
            if inner.value.as_ref() == value {
                return Some(node);
            }

            curr = inner.next;
        }

        None
    }

    #[inline]
    fn hash(&self, key: &T) -> (u64, u8) {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        // H1 = 57 bit Group ID; H2 = 7 bit Fingerprint
        (hash >> 7, (hash & 0x7F) as u8)
    }

    #[inline]
    fn fast_mod(&self, hash: u32) -> u32 {
        ((hash as u64 * self.buckets.len() as u64 - 1) >> 32) as u32
    }
}

impl<T> PartialEq for LinkedSet<T>
where
    T: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().all(|k| other.contains(k))
    }
}

impl<T> Eq for LinkedSet<T> where T: Eq + Hash {}

pub struct Iter<'a, T> {
    node: Option<NonNull<Node<T>>>,
    _marker: PhantomData<&'a Node<T>>,
}

pub struct IntoIter<T> {
    node: Option<NonNull<Node<T>>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let node = self.node?;
        // Safety: The pointer is held by the LinkedSet and this iterator is bound to the
        // lifetime of it so is guaranteed to live that long.
        let inner = unsafe { &*node.as_ptr() };
        self.node = inner.next;
        Some(&inner.value)
    }

    #[inline]
    fn count(self) -> usize {
        self.fold(0, |count, _| count + 1)
    }

    #[inline]
    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x);
        }

        accum
    }
}

impl<T> IntoIterator for LinkedSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        match self.head.take() {
            Some(n) => IntoIter { node: Some(n) },
            None => IntoIter { node: None },
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.node.take() {
            let inner = unsafe { &*node.as_ptr() };
            self.node = inner.next;

            match Rc::try_unwrap(Rc::clone(&inner.value)) {
                Ok(v) => Some(v),
                Err(_) => None,
            }
        } else {
            None
        }
    }
}

impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            _marker: self._marker.clone(),
        }
    }
}

impl<T> Extend<T> for LinkedSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<'a, T> Extend<&'a T> for LinkedSet<T>
where
    T: 'a + Eq + Hash + Copy,
{
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}

impl<T> FromIterator<T> for LinkedSet<T>
where
    T: Eq + Hash,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut ls = LinkedSet::new();
        ls.extend(iter);
        ls
    }
}

impl<T, const N: usize> From<[T; N]> for LinkedSet<T>
where
    T: Eq + Hash,
{
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

pub struct Intersection<'a, T: 'a> {
    iter: Iter<'a, T>,
    other: &'a LinkedSet<T>,
}

pub struct Difference<'a, T: 'a> {
    iter: Iter<'a, T>,
    other: &'a LinkedSet<T>,
}

pub struct SymmetricDifference<'a, T: 'a> {
    iter: Chain<Difference<'a, T>, Difference<'a, T>>,
}

pub struct Union<'a, T: 'a> {
    iter: Chain<Iter<'a, T>, Difference<'a, T>>,
}

impl<T> Clone for Intersection<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            ..*self
        }
    }
}

impl<'a, T> Iterator for Intersection<'a, T>
where
    T: Eq + Hash,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = self.iter.next()?;
            if self.other.contains(next) {
                return Some(next);
            }
        }
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter.fold(init, |acc, elt| {
            if self.other.contains(elt) {
                f(acc, elt)
            } else {
                acc
            }
        })
    }
}

impl<T> Clone for Difference<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            ..*self
        }
    }
}

impl<'a, T> Iterator for Difference<'a, T>
where
    T: Eq + Hash,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = self.iter.next()?;
            if !self.other.contains(next) {
                return Some(next);
            }
        }
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter.fold(init, |acc, elt| {
            if self.other.contains(elt) {
                acc
            } else {
                f(acc, elt)
            }
        })
    }
}

impl<T> Clone for SymmetricDifference<'_, T> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, T> Iterator for SymmetricDifference<'a, T>
where
    T: Eq + Hash,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter.fold(init, f)
    }
}

impl<T> Clone for Union<'_, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, T> Iterator for Union<'a, T>
where
    T: Eq + Hash,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter.fold(init, f)
    }
}

#[cfg(test)]
mod toblerone_test {
    use super::*;

    #[test]
    fn can_add_a_node() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        ls.insert(42);
        assert!(ls.head.is_some());
        assert!(ls.tail.is_some());
        assert_eq!(ls.len(), 1);
        unsafe {
            let head = &*ls.head.unwrap().as_ptr();
            assert_eq!(*head.value, 42);
        }
    }

    #[test]
    fn add_a_lot_of_values() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        assert_eq!(ls.capacity, 16);
        for i in 0..17 {
            ls.insert(i as i32);
        }

        assert_eq!(ls.len(), 17);
        assert_eq!(ls.capacity, 32);
        assert!(ls.head.is_some());
        assert!(ls.tail.is_some());
        unsafe {
            let head = &*ls.head.unwrap().as_ptr();
            assert_eq!(*head.value, 0);
            let tail = &*ls.tail.unwrap().as_ptr();
            assert_eq!(*tail.value, 16);
        }
    }

    #[test]
    fn actually_is_a_set() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for _ in 0..100_000 {
            ls.insert(1);
        }

        assert_eq!(ls.len(), 1);
    }

    #[test]
    fn get_some_value() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..300_000 {
            ls.insert(i);
        }

        let item = ls.get(&123456);
        assert!(item.is_some());
        assert_eq!(*item.unwrap(), 123456);
    }

    #[test]
    fn iterator() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..17 {
            ls.insert(i as i32);
        }

        for (item, expected) in ls.iter().zip((0..17).into_iter()) {
            assert_eq!(*item, expected);
        }
    }

    #[test]
    fn into_iterator() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..100 {
            ls.insert(i);
        }

        for (item, check) in ls.into_iter().zip((0..100).into_iter()) {
            assert_eq!(item, check);
        }
    }

    #[test]
    fn remove_node_head() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..20 {
            ls.insert(i as i32);
        }

        assert!(ls.remove(&0));
        unsafe {
            let head = &*ls.head.unwrap().as_ptr();
            assert_eq!(*head.value, 1);

            let tail = &*ls.tail.unwrap().as_ptr();
            assert_eq!(*tail.value, 19);
        }
    }

    #[test]
    fn remove_node_tail() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..20 {
            ls.insert(i as i32);
        }

        assert!(ls.remove(&19));

        unsafe {
            let head = &*ls.head.unwrap().as_ptr();
            assert_eq!(*head.value, 0);

            let tail = &*ls.tail.unwrap().as_ptr();
            assert_eq!(*tail.value, 18);
        }
    }

    #[test]
    fn remove_node_middle() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..20 {
            ls.insert(i as i32);
        }

        assert!(ls.remove(&14));
        assert_eq!(ls.get(&14), None);
    }

    #[test]
    fn clear() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..100_000 {
            ls.insert(i as i32);
        }

        ls.clear();
        assert!(ls.head.is_none());
        assert!(ls.tail.is_none());
        assert_eq!(ls.len(), 0);
    }

    #[test]
    fn reuse() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..100_000 {
            ls.insert(i as i32);
            println!("{i}");
        }

        assert_eq!(ls.len(), 100_000);

        ls.clear();
        assert!(ls.head.is_none());
        assert!(ls.tail.is_none());
        assert_eq!(ls.len(), 0);

        ls.insert(100);
        assert_eq!(ls.len(), 1);

        unsafe {
            assert!(ls.head.is_some());
            let head = &*ls.head.unwrap().as_ptr();
            assert_eq!(*head.value, 100);

            assert!(ls.tail.is_some());
            let tail = &*ls.tail.unwrap().as_ptr();
            assert_eq!(*tail.value, 100);
        }
    }

    #[test]
    fn disjoint() {
        let a = LinkedSet::from([1, 2, 3]);
        let mut b = LinkedSet::new();

        assert!(a.is_disjoint(&b));
        b.insert(4);
        assert!(a.is_disjoint(&b));
        b.insert(1);
        assert_eq!(a.is_disjoint(&b), false);
    }

    #[test]
    fn subset() {
        let a = LinkedSet::from([1, 2, 3]);
        let mut b = LinkedSet::new();
        assert!(b.is_subset(&a));
        b.insert(2);
        assert!(b.is_subset(&a));
        b.insert(4);
        assert_eq!(b.is_subset(&a), false);
    }

    #[test]
    fn superset() {
        let a = LinkedSet::from([1, 2]);
        let mut b = LinkedSet::new();

        assert_eq!(b.is_superset(&a), false);

        b.insert(0);
        b.insert(1);
        assert_eq!(b.is_superset(&a), false);

        b.insert(2);
        assert!(b.is_superset(&a));
    }

    #[test]
    fn extend() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        ls.extend([1, 2, 3].iter());

        assert_eq!(ls.len(), 3);
        for (item, expected) in ls.iter().zip([1, 2, 3].iter()) {
            assert_eq!(*item, *expected);
        }
    }

    #[test]
    fn difference() {
        let a = LinkedSet::from([1, 2, 3]);
        let b = LinkedSet::from([4, 2, 3, 4]);

        let diff: LinkedSet<_> = a.difference(&b).collect();
        assert_eq!(diff, [1].iter().collect());

        let diff: LinkedSet<_> = b.difference(&a).collect();
        assert_eq!(diff, [4].iter().collect());
    }

    #[test]
    fn symmetric_difference() {
        let a = LinkedSet::from([1, 2, 3]);
        let b = LinkedSet::from([4, 2, 3, 4]);

        let diff1: LinkedSet<_> = a.symmetric_difference(&b).collect();
        let diff2: LinkedSet<_> = b.symmetric_difference(&a).collect();

        assert_eq!(diff1, diff2);
        assert_eq!(diff1, [1, 4].iter().collect());
    }

    #[test]
    fn intersection() {
        let a = LinkedSet::from([1, 2, 3]);
        let b = LinkedSet::from([4, 2, 3, 4]);

        let intersection: LinkedSet<_> = a.intersection(&b).collect();
        assert_eq!(intersection, [2, 3].iter().collect());
    }

    #[test]
    fn union() {
        let a = LinkedSet::from([1, 2, 3]);
        let b = LinkedSet::from([4, 2, 3, 4]);

        let union: LinkedSet<_> = a.union(&b).collect();
        assert_eq!(union, [1, 2, 3, 4].iter().collect());
    }
}
