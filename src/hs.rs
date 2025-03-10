use std::{
    collections::HashSet, hash::Hash, iter::Chain, marker::PhantomData, ptr::NonNull, rc::Rc,
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
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            node: self.head,
            _marker: std::marker::PhantomData,
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
mod hs_tests {
    use super::*;

    #[test]
    fn can_add_nodes() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        for i in 0..100_000 {
            ls.insert(i);
        }

        assert_eq!(ls.size, 100_000);
    }
}
