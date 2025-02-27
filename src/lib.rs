use std::{
    cell::RefCell,
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    rc::{Rc, Weak},
};

#[cfg(target_arch = "x86")]
use core::arch::x86 as cpu;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as cpu;

const DEFAULT_CAPACITY: usize = 16;
const BUCKET_SIZE: usize = 16;

const EMPTY: u8 = 0x80;
const TOMB: u8 = 0xFE;

type Link<T> = Option<Rc<Node<T>>>;
type WeakLink<T> = Option<Weak<Node<T>>>;

#[derive(Debug)]
struct Node<T> {
    pub value: T,
    pub prev: RefCell<WeakLink<T>>,
    pub next: RefCell<Link<T>>,
}

impl<T> Node<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            prev: RefCell::new(None),
            next: RefCell::new(None),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
struct Bucket<T> {
    meta: [u8; BUCKET_SIZE],
    slots: Vec<Link<T>>,
}

impl<T> Bucket<T> {
    pub fn new() -> Self {
        Self {
            meta: [EMPTY; BUCKET_SIZE],
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
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            head: None,
            tail: None,
            buckets: Vec::new(),
            capacity: DEFAULT_CAPACITY,
            size: 0,
        }
    }

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

        let (h1, h2) = self.hash(&value);
        let node = Rc::new(Node::new(value));

        let mut gidx = self.fast_mod(h1 as u32);
        loop {
            if gidx as usize == self.buckets.len() {
                return false;
            }

            let bucket = &mut self.buckets[gidx as usize];
            match bucket.simd_free_slot() {
                Some(idx) => {
                    let bucket_node = Rc::clone(&node);
                    bucket.slots[idx] = Some(bucket_node);
                    bucket.meta[idx] = h2;
                    self.add_node(Rc::clone(&node));
                    self.size += 1;
                    return true;
                }
                None => {
                    gidx += 1;
                }
            }
        }
    }

    #[inline]
    pub fn get(&self, value: &T) -> Option<&T> {
        let (h1, h2) = self.hash(&value);
        let mut gidx = self.fast_mod(h1 as u32);
        loop {
            if gidx as usize == self.buckets.len() {
                return None;
            }
            let bucket = &self.buckets[gidx as usize];
            let potentials = bucket.simd_hash_match(h2);
            for slot in potentials {
                match bucket.slots[slot] {
                    Some(ref node) => {
                        if &node.value != value {
                            gidx += 1;
                            continue;
                        }

                        return Some(&node.value);
                    }
                    None => {}
                }
            }

            gidx += 1;
        }
    }

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

        // TODO: Maybe extract this out into a helper?
        loop {
            if gidx as usize == self.buckets.len() {
                return false;
            }

            let bucket = &mut self.buckets[gidx as usize];
            let potentials = bucket.simd_hash_match(h2);
            for slot in potentials.iter() {
                match bucket.slots[*slot] {
                    Some(ref node) => {
                        if &node.value != value {
                            gidx += 1;
                            continue;
                        }

                        bucket.meta[*slot] = TOMB;
                        self.size -= 1;
                        return true;
                    }
                    None => {}
                }
            }

            gidx += 1;
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn clear(&mut self) {
        if self.buckets.is_empty() {
            return;
        }

        // Probably need some custom dropping logic here but meh
        // it's just RCs all the way down
        self.head = None;
        self.tail = None;
        self.buckets = Vec::with_capacity(self.capacity);
    }

    pub fn iter(&self) -> LinkedSetIterator<'_, T> {
        let head = match self.head {
            Some(ref node) => Some(Rc::clone(&node)),
            None => None,
        };

        LinkedSetIterator {
            node: head,
            _marker: std::marker::PhantomData,
        }
    }

    fn should_resize(&self) -> bool {
        self.buckets.is_empty() || self.size > 4 * self.capacity / 5
    }

    fn add_node(&mut self, node: Rc<Node<T>>) {
        if self.head.is_none() {
            self.head = Some(Rc::clone(&node));
            self.tail = Some(node);
        } else {
            let tail = self.tail.take().expect("if head is set then so is tail");
            *tail.next.borrow_mut() = Some(Rc::clone(&node));
            *node.prev.borrow_mut() = Some(Rc::downgrade(&tail));
            self.tail = Some(node);
        }
    }

    pub(crate) fn insert_with_node(&mut self, node: Rc<Node<T>>) {
        let (h1, h2) = self.hash(&node.value);
        let mut gidx = self.fast_mod(h1 as u32);

        loop {
            let bucket = &mut self.buckets[gidx as usize];
            match bucket.simd_free_slot() {
                Some(idx) => {
                    let bucket_node = Rc::clone(&node);
                    bucket.slots[idx] = Some(bucket_node);
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

    fn remove_node(&mut self, node: Rc<Node<T>>) {
        // If the node is the head
        if self.head.as_mut().unwrap().value == node.value {
            let next = node.next.borrow_mut().take();
            if let Some(ref n) = next {
                n.prev.borrow_mut().take();
            }
            self.head = next;

            return;
        }

        // If the node is the tail
        if self.tail.as_mut().unwrap().value == node.value {
            let prev = node.prev.borrow_mut().take();
            assert!(prev.is_some());
            self.tail = Weak::upgrade(&prev.unwrap());
            *self.tail.as_mut().unwrap().next.borrow_mut() = None;
            return;
        }

        // If the node is the middle
        let next = node.next.borrow_mut().take();
        let prev = node.prev.borrow_mut().take();
        assert!(next.is_some());
        assert!(prev.is_some());

        if let Some(next_ptr) = next.as_ref().and_then(|n| Some(n.clone())) {
            *next_ptr.prev.borrow_mut() = prev.clone();
        }

        if let Some(prev_ptr) = prev.and_then(|weak_ref| Weak::upgrade(&weak_ref)) {
            *prev_ptr.next.borrow_mut() = next.clone();
        }

        // prev  node  next
        //  ^           ^
        //   \         /
        //     -------
        //    <--//-->

        // let mut prev_strong = Weak::upgrade(&prev.unwrap());
        // *prev_strong.as_mut().unwrap().next.borrow_mut() = Some(Rc::clone(&next.unwrap()));
        // *next.as_mut().unwrap().prev.borrow_mut() = prev;
    }

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
                        new_list.insert_with_node(n);
                    }
                    None => {}
                }
            }
        }

        if let Some(head) = &self.head {
            new_list.head = Some(Rc::clone(&head));
            new_list.tail = Some(Rc::clone(
                &self
                    .tail
                    .as_ref()
                    .expect("head is set which means tail is set"),
            ));
        }
        *self = new_list;
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

pub struct LinkedSetIterator<'a, T> {
    node: Option<Rc<Node<T>>>,
    _marker: PhantomData<&'a Node<T>>,
}

impl<'a, T> Iterator for LinkedSetIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.node.as_ref()?;
        // Safety: The pointer is held by the LinkedSet and this iterator is bound to the
        // lifetime of it so is guaranteed to live that long.
        let inner = unsafe { &*Rc::as_ptr(node) };
        self.node = inner.next.borrow().as_ref().map(|n| Rc::clone(&n));
        Some(&inner.value)
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
        let head = ls.head.as_ref().unwrap();
        assert_eq!(head.value, 42);
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
        let head = ls.head.as_ref().unwrap();
        assert_eq!(head.value, 0);

        let tail = ls.tail.as_ref().unwrap();
        assert_eq!(tail.value, 16);
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
}
