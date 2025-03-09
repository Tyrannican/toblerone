use std::{collections::HashSet, hash::Hash, ptr::NonNull, rc::Rc};

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
