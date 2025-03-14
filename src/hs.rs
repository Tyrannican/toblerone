use std::{
    collections::{HashSet, TryReserveError},
    hash::Hash,
    iter::Chain,
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

#[derive(Debug)]
struct Node<T> {
    next: Option<NonNull<Node<T>>>,
    prev: Option<NonNull<Node<T>>>,
    value: Rc<T>,
}

impl<T> Node<T> {
    pub fn new(value: T) -> Self {
        Self {
            next: None,
            prev: None,
            value: Rc::new(value),
        }
    }
}

#[derive(Debug)]
pub struct LinkedSet<T> {
    inner: HashSet<Rc<T>>,
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    size: usize,
}

impl<T> LinkedSet<T>
where
    T: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            inner: HashSet::default(),
            head: None,
            tail: None,
            size: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: HashSet::with_capacity(capacity),
            head: None,
            tail: None,
            size: 0,
        }
    }

    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        if self.contains(&value) {
            return false;
        }

        let node = Box::new(Node::new(value));
        self.inner.insert(Rc::clone(&node.value));
        self.add_node(node);
        self.size += 1;

        true
    }

    #[inline]
    pub fn get<'a>(&'a self, value: &'a T) -> Option<&'a T> {
        match self.inner.get(value) {
            Some(v) => Some(v.as_ref()),
            None => None,
        }
    }

    #[inline]
    pub fn remove(&mut self, value: &T) -> bool {
        if !self.contains(value) {
            return false;
        }

        self.inner.remove(value);
        self.remove_node(value);
        self.size -= 1;

        true
    }

    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.get(value).is_some()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
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
        self.inner.clear();
        self.head = None;
        self.tail = None;
        self.size = 0
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity);
    }

    #[inline]
    pub fn replace(&mut self, value: T) -> Option<T> {
        if !self.contains(&value) {
            self.insert(value);
            return None;
        }

        // Fugly AF
        let mut curr = self.head;
        while let Some(curr_node) = curr {
            let n_inner = unsafe { &*curr_node.as_ptr() };
            if *n_inner.value == value {
                let new_node = NonNull::new(Box::leak(Box::new(Node::new(value))));
                unsafe {
                    let node = &mut *new_node.unwrap().as_mut();
                    node.prev = n_inner.prev;
                    node.next = n_inner.next;
                    if let Some(mut prev) = n_inner.prev {
                        let p_inner = &mut *prev.as_mut();
                        let _ = std::mem::replace(&mut node.prev, n_inner.prev);
                        let _ = std::mem::replace(&mut p_inner.next, new_node);
                    }

                    if let Some(mut next) = n_inner.next {
                        let next_inner = &mut *next.as_mut();
                        let _ = std::mem::replace(&mut node.next, n_inner.next);
                        let _ = std::mem::replace(&mut next_inner.prev, new_node);
                    }
                }

                if curr == self.head {
                    self.head = new_node;
                }
                if curr == self.tail {
                    self.tail = new_node;
                }

                self.inner.remove(&n_inner.value);
                let ptr = unsafe { Box::from_raw(curr_node.as_ptr()) };
                return Rc::into_inner(ptr.value);
            }
            curr = n_inner.next;
        }

        None
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        todo!()
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        todo!()
    }

    #[inline]
    pub fn take(&mut self, value: &T) {
        todo!()
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            node: self.head,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn drain(&mut self) -> Drain<'_, T> {
        Drain {
            set: self,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let to_remove = self
            .inner
            .iter()
            .filter_map(|item| {
                if !f(item) {
                    return Some(Rc::clone(item));
                }

                None
            })
            .collect::<Vec<Rc<T>>>();

        for item in to_remove {
            self.remove(&item);
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
    fn remove_node(&mut self, value: &T) {
        // If the node is the head
        if let Some(head) = self.head {
            let h_inner = unsafe { &*head.as_ptr() };
            if h_inner.value.as_ref() == value {
                self.head = h_inner.next;
                return;
            }
        }

        if let Some(tail) = self.tail {
            // If the node is the tail
            let t_inner = unsafe { &*tail.as_ptr() };
            if t_inner.value.as_ref() == value {
                self.tail = t_inner.prev;
                return;
            }
        }

        // Node is in the middle - find and remove
        let mut curr = self.head;
        while let Some(node) = curr {
            let inner = unsafe { &*node.as_ptr() };
            if inner.value.as_ref() == value {
                let next = inner.next;
                let prev = inner.prev;

                assert!(next.is_some());
                assert!(prev.is_some());

                unsafe {
                    (*prev.unwrap().as_ptr()).next = next;
                    (*next.unwrap().as_ptr()).prev = prev;
                }
            }

            curr = inner.next;
        }
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
            // Safety: We just took ownership of it
            let inner = unsafe { Box::from_raw(node.as_ptr()) };
            self.node = inner.next;

            match Rc::try_unwrap(inner.value) {
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

pub struct Drain<'a, T> {
    set: &'a mut LinkedSet<T>,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for Drain<'a, T>
where
    T: Eq + Hash,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match self.set.head.take() {
            Some(head) => {
                // Safety: We just took ownership of it
                let head = unsafe { Box::from_raw(head.as_ptr()) };
                self.set.inner.remove(&head.value);
                match Rc::into_inner(head.value) {
                    Some(value) => {
                        self.set.head = head.next;
                        self.set.size -= 1;
                        return Some(value);
                    }
                    None => None,
                }
            }
            None => None,
        }
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
mod hs_tests {
    use super::*;

    #[test]
    fn capacity() {
        let ls: LinkedSet<i32> = LinkedSet::with_capacity(100);
        assert!(ls.capacity() >= 100);
    }

    #[test]
    fn can_add_nodes() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..100_000 {
            ls.insert(i);
        }

        assert_eq!(ls.size, 100_000);
    }

    #[test]
    fn get_a_value() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..300_000 {
            ls.insert(i);
        }

        let item = ls.get(&123456);
        assert!(item.is_some());
        assert_eq!(item, Some(&123456));
    }

    #[test]
    fn ordering() {
        let mut ls = LinkedSet::from([1, 2, 3]);
        for (item, n) in ls.iter().zip([1, 2, 3].iter()) {
            assert_eq!(item, n);
        }

        ls.remove(&2);
        for (item, n) in ls.iter().zip([1, 3].iter()) {
            assert_eq!(item, n);
        }
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
    fn retain() {
        let mut hs = LinkedSet::from([1, 2, 3, 4, 5, 6]);
        hs.retain(|k| k % 2 == 0);
        assert_eq!(hs, LinkedSet::from([2, 4, 6]));
    }

    #[test]
    fn remove_node_head() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..20 {
            ls.insert(i as i32);
        }

        assert!(ls.remove(&0));
        assert_eq!(ls.len(), 19);

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
        assert_eq!(ls.len(), 19);

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
        assert!(ls.is_empty());
        assert!(ls.head.is_none());
        assert!(ls.tail.is_none());
    }

    #[test]
    fn replace() {
        todo!()
    }

    #[test]
    fn shrink_to_fit() {
        let mut ls: LinkedSet<i32> = LinkedSet::with_capacity(100);
        ls.insert(1);
        ls.insert(2);
        assert!(ls.capacity() >= 100);
        ls.shrink_to_fit();
        assert!(ls.capacity() >= 2);
    }

    #[test]
    fn shrink_to() {
        let mut ls: LinkedSet<i32> = LinkedSet::with_capacity(100);
        ls.insert(1);
        ls.insert(2);
        assert!(ls.capacity() >= 100);
        ls.shrink_to(10);
        assert!(ls.capacity() >= 10);
        ls.shrink_to(0);
        assert!(ls.capacity() >= 2);
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

    #[test]
    fn drain() {
        let mut ls = LinkedSet::from([1, 2, 3]);
        assert!(!ls.is_empty());

        for _ in ls.drain() {}
        assert!(ls.is_empty());
    }
}
