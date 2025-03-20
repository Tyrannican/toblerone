use std::{
    borrow::Borrow,
    collections::{HashSet, TryReserveError},
    hash::Hash,
    iter::Chain,
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

#[derive(Eq, PartialEq, Hash, Debug)]
struct InnerValue<T> {
    value: Rc<T>,
}

impl<T> InnerValue<T>
where
    T: Eq + Hash,
{
    pub fn new(value: T) -> Self {
        Self {
            value: Rc::new(value),
        }
    }

    pub fn consume(self) -> Option<T> {
        Rc::into_inner(self.value)
    }
}

impl<T> AsRef<T> for InnerValue<T> {
    fn as_ref(&self) -> &T {
        self.value.as_ref()
    }
}

impl<T> Borrow<T> for InnerValue<T>
where
    T: Eq + Hash,
{
    fn borrow(&self) -> &T {
        self.value.as_ref()
    }
}

impl<T> Clone for InnerValue<T>
where
    T: Eq + Hash,
{
    fn clone(&self) -> Self {
        Self {
            value: Rc::clone(&self.value),
        }
    }
}

#[derive(Debug)]
struct Node<T> {
    next: Option<NonNull<Node<T>>>,
    prev: Option<NonNull<Node<T>>>,
    value: InnerValue<T>,
}

impl<T> Node<T>
where
    T: Eq + Hash,
{
    pub fn new(value: T) -> Self {
        Self {
            next: None,
            prev: None,
            value: InnerValue::new(value),
        }
    }
}

// TODO: Implement ability to pass in custom Hasher
// TODO: Fix doc-related issues in implementation

/// A [hash set] which wraps around a [`std::collections::HashSet`] with an underlying
/// Linked List so maintain ordering.
///
/// This is not a complete wrapper around the Standard Library Hash Set as some utility
/// is still missing but will be added as time goes one.
///
/// It behaves exactly like [`std::collections::HashSet`] with only a slight hit in performance
/// due to also maintaining the internal Linked List.
///
/// # Examples
///
/// ```
/// use toblerone::LinkedSet;
/// // Type inference lets us omit an explicit type signature (which
/// // would be `LinkedSet<String>` in this example).
/// let mut books = LinkedSet::new();
///
/// // Add some books.
/// books.insert("A Dance With Dragons".to_string());
/// books.insert("To Kill a Mockingbird".to_string());
/// books.insert("The Odyssey".to_string());
/// books.insert("The Great Gatsby".to_string());
///
/// // Check for a specific one.
/// if !books.contains("The Winds of Winter") {
///     println!("We have {} books, but The Winds of Winter ain't one.",
///              books.len());
/// }
///
/// // Remove a book.
/// books.remove("The Odyssey");
///
/// // Iterate over everything.
/// for book in &books {
///     println!("{book}");
/// }
/// ```
///
/// The easiest way to use `LinkedSet` with a custom type is to derive
/// [`Eq`] and [`Hash`]. We must also derive [`PartialEq`],
/// which is required if [`Eq`] is derived.
///
/// ```
/// use toblerone::LinkedSet;
/// #[derive(Hash, Eq, PartialEq, Debug)]
/// struct Viking {
///     name: String,
///     power: usize,
/// }
///
/// let mut vikings = LinkedSet::new();
///
/// vikings.insert(Viking { name: "Einar".to_string(), power: 9 });
/// vikings.insert(Viking { name: "Einar".to_string(), power: 9 });
/// vikings.insert(Viking { name: "Olaf".to_string(), power: 4 });
/// vikings.insert(Viking { name: "Harald".to_string(), power: 8 });
///
/// // Use derived implementation to print the vikings.
/// for x in &vikings {
///     println!("{x:?}");
/// }
/// ```
///
/// A `LinkedSet` with a known list of items can be initialized from an array:
///
/// ```
/// use toblerone::LinkedSet;
///
/// let viking_names = LinkedSet::from(["Einar", "Olaf", "Harald"]);
/// ```
#[derive(Debug)]
pub struct LinkedSet<T> {
    inner: HashSet<InnerValue<T>>,
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    size: usize,
}

impl<T> LinkedSet<T>
where
    T: Eq + Hash,
{
    /// Creates an empty `LinkedSet`.
    ///
    /// The linked set is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let set: LinkedSet<i32> = LinkedSet::new();
    /// ```
    pub fn new() -> Self {
        Self {
            inner: HashSet::default(),
            head: None,
            tail: None,
            size: 0,
        }
    }

    /// Creates an empty `LinkedSet` with at least the specified capacity.
    ///
    /// The hash set will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the hash set will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let set: LinkedSet<i32> = LinkedSet::with_capacity(10);
    /// assert!(set.capacity() >= 10);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: HashSet::with_capacity(capacity),
            head: None,
            tail: None,
            size: 0,
        }
    }

    /// Adds a value to the set.
    ///
    /// Returns whether the value was newly inserted. That is:
    ///
    /// - If the set did not previously contain this value, `true` is returned.
    /// - If the set already contained this value, `false` is returned,
    ///   and the set is not modified: original value is not replaced,
    ///   and the value passed as argument is dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut set = LinkedSet::new();
    ///
    /// assert_eq!(set.insert(2), true);
    /// assert_eq!(set.insert(2), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, value: T) -> bool {
        if self.contains(&value) {
            return false;
        }

        let node = Box::new(Node::new(value));
        self.inner.insert(node.value.clone());
        self.add_node(node);
        self.size += 1;

        true
    }

    /// Returns a reference to the value in the set, if any, that is equal to the given value.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let set = LinkedSet::from([1, 2, 3]);
    /// assert_eq!(set.get(&2), Some(&2));
    /// assert_eq!(set.get(&4), None);
    /// ```
    #[inline]
    pub fn get<'a>(&'a self, value: &'a T) -> Option<&'a T> {
        match self.inner.get(value) {
            Some(v) => Some(v.as_ref()),
            None => None,
        }
    }

    /// Removes a value from the set. Returns whether the value was
    /// present in the set.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut set = LinkedSet::new();
    ///
    /// set.insert(2);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
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

    /// Returns `true` if the set contains a value.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let set = LinkedSet::from([1, 2, 3]);
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.get(value).is_some()
    }

    /// Returns the number of elements the set can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let set: LinkedSet<i32> = LinkedSet::with_capacity(100);
    /// assert!(set.capacity() >= 100);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns the number of elements in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut v = LinkedSet::new();
    /// assert_eq!(v.len(), 0);
    /// v.insert(1);
    /// assert_eq!(v.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the set contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut v = LinkedSet::new();
    /// assert!(v.is_empty());
    /// v.insert(1);
    /// assert!(!v.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clears the set, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut v = LinkedSet::new();
    /// v.insert(1);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
        self.head = None;
        self.tail = None;
        self.size = 0
    }

    /// Shrinks the capacity of the set as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut set = LinkedSet::with_capacity(100);
    /// set.insert(1);
    /// set.insert(2);
    /// assert!(set.capacity() >= 100);
    /// set.shrink_to_fit();
    /// assert!(set.capacity() >= 2);
    /// ```
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// Shrinks the capacity of the set with a lower limit. It will drop
    /// down no lower than the supplied limit while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut set = LinkedSet::with_capacity(100);
    /// set.insert(1);
    /// set.insert(2);
    /// assert!(set.capacity() >= 100);
    /// set.shrink_to(10);
    /// assert!(set.capacity() >= 10);
    /// set.shrink_to(0);
    /// assert!(set.capacity() >= 2);
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity);
    }

    // TODO: Implement `replace` at some point

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `LinkedSet`. The collection may reserve more space to speculatively
    /// avoid frequent reallocations. After calling `reserve`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let mut set: LinkedSet<i32> = LinkedSet::new();
    /// set.reserve(10);
    /// assert!(set.capacity() >= 10);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted
    /// in the `LinkedSet`. The collection may reserve more space to speculatively
    /// avoid frequent reallocations. After calling `try_reserve`,
    /// capacity will be greater than or equal to `self.len() + additional` if
    /// it returns `Ok(())`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let mut set: LinkedSet<i32> = LinkedSet::new();
    /// set.try_reserve(10).expect("why is the test harness OOMing on a handful of bytes?");
    /// ```
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
    ///
    /// The value may be any borrowed form of the set's value type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the value type.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut set = LinkedSet::from([1, 2, 3]);
    /// assert_eq!(set.take(&2), Some(2));
    /// assert_eq!(set.take(&2), None);
    /// ```
    #[inline]
    pub fn take(&mut self, value: &T) -> Option<T> {
        if !self.contains(value) {
            return None;
        }

        self.inner.remove(value);
        let mut curr = self.head;
        while let Some(mut curr_node) = curr {
            // Safety: The node is guaranteed to exist here
            let n_inner = unsafe { &mut *curr_node.as_mut() };
            if &*n_inner.value.as_ref() != value {
                curr = n_inner.next;
                continue;
            }

            // Safety: The nodes are guaranteed to exist so we can take the ptr
            unsafe {
                if let Some(mut prev) = n_inner.prev {
                    let p_inner = &mut *prev.as_mut();
                    p_inner.next = n_inner.next;
                }

                if let Some(mut next) = n_inner.next {
                    let next_inner = &mut *next.as_mut();
                    next_inner.prev = n_inner.prev;
                }
                n_inner.prev = None;
                n_inner.next = None;

                self.size -= 1;
                let ptr = Box::from_raw(curr_node.as_ptr());
                return ptr.value.consume();
            }
        }

        None
    }

    /// An iterator visiting all elements in arbitrary order.
    /// The iterator element type is `&'a T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let mut set = LinkedSet::new();
    /// set.insert("a");
    /// set.insert("b");
    ///
    /// // Will print in an arbitrary order.
    /// for x in set.iter() {
    ///     println!("{x}");
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// This takes O(n) time instead of the stdlib O(capacity) time as we're
    /// iterating through the Linked List instead of the Buckets in the underlying
    /// Hashset
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            node: self.head,
            _marker: std::marker::PhantomData,
        }
    }

    /// Clears the set, returning all elements as an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut set = LinkedSet::from([1, 2, 3]);
    /// assert!(!set.is_empty());
    ///
    /// // print 1, 2, 3 in an arbitrary order
    /// for i in set.drain() {
    ///     println!("{i}");
    /// }
    ///
    /// assert!(set.is_empty());
    /// ```
    #[inline]
    pub fn drain(&mut self) -> Drain<'_, T> {
        Drain {
            set: self,
            _marker: std::marker::PhantomData,
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// The elements are visited in unsorted (and unspecified) order.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let mut set = LinkedSet::from([1, 2, 3, 4, 5, 6]);
    /// set.retain(|&k| k % 2 == 0);
    /// assert_eq!(set, LinkedSet::from([2, 4, 6]));
    /// ```
    ///
    /// # Performance
    ///
    /// This implementation takes O(n) time instead of the stdlib O(capacity) time
    /// as we use the Linked List for the iteration.
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let to_remove = self
            .inner
            .iter()
            .filter_map(|item| {
                if !f(item.as_ref()) {
                    return Some(item.clone());
                }

                None
            })
            .collect::<Vec<InnerValue<T>>>();

        for item in to_remove {
            self.remove(&item.as_ref());
        }
    }

    /// Returns `true` if `self` has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let a = LinkedSet::from([1, 2, 3]);
    /// let mut b = LinkedSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    #[inline]
    pub fn is_disjoint(&self, other: &LinkedSet<T>) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Returns `true` if the set is a subset of another,
    /// i.e., `other` contains at least all the values in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let sup = LinkedSet::from([1, 2, 3]);
    /// let mut set = LinkedSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    #[inline]
    pub fn is_subset(&self, other: &LinkedSet<T>) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| other.contains(v))
        } else {
            false
        }
    }

    /// Returns `true` if the set is a superset of another,
    /// i.e., `self` contains at least all the values in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    ///
    /// let sub = LinkedSet::from([1, 2]);
    /// let mut set = LinkedSet::new();
    ///
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(2);
    /// assert_eq!(set.is_superset(&sub), true);
    /// ```
    #[inline]
    pub fn is_superset(&self, other: &LinkedSet<T>) -> bool {
        other.is_subset(&self)
    }

    /// Visits the values representing the difference,
    /// i.e., the values that are in `self` but not in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let a = LinkedSet::from([1, 2, 3]);
    /// let b = LinkedSet::from([4, 2, 3, 4]);
    ///
    /// // Can be seen as `a - b`.
    /// for x in a.difference(&b) {
    ///     println!("{x}"); // Print 1
    /// }
    ///
    /// let diff: LinkedSet<_> = a.difference(&b).collect();
    /// assert_eq!(diff, [1].iter().collect());
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else:
    /// let diff: LinkedSet<_> = b.difference(&a).collect();
    /// assert_eq!(diff, [4].iter().collect());
    /// ```
    #[inline]
    pub fn difference<'a>(&'a self, other: &'a LinkedSet<T>) -> Difference<'a, T> {
        Difference {
            iter: self.iter(),
            other,
        }
    }

    /// Visits the values representing the symmetric difference,
    /// i.e., the values that are in `self` or in `other` but not in both.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let a = LinkedSet::from([1, 2, 3]);
    /// let b = LinkedSet::from([4, 2, 3, 4]);
    ///
    /// // Print 1, 4 in arbitrary order.
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{x}");
    /// }
    ///
    /// let diff1: LinkedSet<_> = a.symmetric_difference(&b).collect();
    /// let diff2: LinkedSet<_> = b.symmetric_difference(&a).collect();
    ///
    /// assert_eq!(diff1, diff2);
    /// assert_eq!(diff1, [1, 4].iter().collect());
    /// ```
    #[inline]
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a LinkedSet<T>,
    ) -> SymmetricDifference<'a, T> {
        SymmetricDifference {
            iter: self.difference(other).chain(other.difference(self)),
        }
    }

    /// Visits the values representing the intersection,
    /// i.e., the values that are both in `self` and `other`.
    ///
    /// When an equal element is present in `self` and `other`
    /// then the resulting `Intersection` may yield references to
    /// one or the other. This can be relevant if `T` contains fields which
    /// are not compared by its `Eq` implementation, and may hold different
    /// value between the two equal copies of `T` in the two sets.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let a = LinkedSet::from([1, 2, 3]);
    /// let b = LinkedSet::from([4, 2, 3, 4]);
    ///
    /// // Print 2, 3 in arbitrary order.
    /// for x in a.intersection(&b) {
    ///     println!("{x}");
    /// }
    ///
    /// let intersection: LinkedSet<_> = a.intersection(&b).collect();
    /// assert_eq!(intersection, [2, 3].iter().collect());
    /// ```
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

    /// Visits the values representing the union,
    /// i.e., all the values in `self` or `other`, without duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// use toblerone::LinkedSet;
    /// let a = LinkedSet::from([1, 2, 3]);
    /// let b = LinkedSet::from([4, 2, 3, 4]);
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order.
    /// for x in a.union(&b) {
    ///     println!("{x}");
    /// }
    ///
    /// let union: LinkedSet<_> = a.union(&b).collect();
    /// assert_eq!(union, [1, 2, 3, 4].iter().collect());
    /// ```
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

    /// Adds an item to the internal Linked List
    #[inline]
    fn add_node(&mut self, node: Box<Node<T>>) {
        let node = NonNull::new(Box::leak(node));
        if self.head.is_none() {
            self.head = node;
            self.tail = node;
        } else {
            let tail = self.tail.take().expect("if head is set then so is tail");

            // Safety: The nodes are guaranteed to exist
            unsafe {
                (*tail.as_ptr()).next = node;
                (*node.expect("this is guaranteed to be non-null").as_ptr()).prev = Some(tail);
            }

            self.tail = node;
        }
    }

    /// Removes a node from the internal Linked List
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

                // Safety: The nodes are guaranteed to exist
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

/// An iterator over the items of a `HashSet`.
///
/// This `struct` is created by the [`iter`] method on [`HashSet`].
/// See its documentation for more.
///
/// [`iter`]: LinkedSet::iter
///
/// # Examples
///
/// ```
/// use toblerone::LinkedSet;
///
/// let a = LinkedSet::from([1, 2, 3]);
///
/// let mut iter = a.iter();
/// ```
pub struct Iter<'a, T> {
    node: Option<NonNull<Node<T>>>,
    _marker: PhantomData<&'a Node<T>>,
}

/// An owning iterator over the items of a `LinkedSet`.
///
/// This `struct` is created by the [`into_iter`] method on [`LinkedSet`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: IntoIterator::into_iter
///
/// # Examples
///
/// ```
/// use toblerone::LinkedSet;
///
/// let a = LinkedSet::from([1, 2, 3]);
///
/// let mut iter = a.into_iter();
/// ```
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
        Some(inner.value.as_ref())
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

impl<T> IntoIterator for LinkedSet<T>
where
    T: Eq + Hash,
{
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        match self.head.take() {
            Some(n) => IntoIter { node: Some(n) },
            None => IntoIter { node: None },
        }
    }
}

impl<T> Iterator for IntoIter<T>
where
    T: Eq + Hash,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.node.take() {
            // Safety: We just took ownership of it
            let inner = unsafe { Box::from_raw(node.as_ptr()) };
            self.node = inner.next;

            inner.value.consume()
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

/// A draining iterator over the items of a `LinkedSet`.
///
/// This `struct` is created by the [`drain`] method on [`LinkedSet`].
/// See its documentation for more.
///
/// [`drain`]: LinkedSet::drain
///
/// # Examples
///
/// ```
/// use toblerone::LinkedSet;
///
/// let mut a = LinkedSet::from([1, 2, 3]);
///
/// let mut drain = a.drain();
/// ```
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
                match head.value.consume() {
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

/// A lazy iterator producing elements in the intersection of `LinkedSet`s.
///
/// This `struct` is created by the [`intersection`] method on [`LinkedSet`].
/// See its documentation for more.
///
/// [`intersection`]: LinkedSet::intersection
///
/// # Examples
///
/// ```
/// use toblerone::LinkedSet;
///
/// let a = LinkedSet::from([1, 2, 3]);
/// let b = LinkedSet::from([4, 2, 3, 4]);
///
/// let mut intersection = a.intersection(&b);
/// ```
pub struct Intersection<'a, T: 'a> {
    iter: Iter<'a, T>,
    other: &'a LinkedSet<T>,
}

/// A lazy iterator producing elements in the difference of `LinkedSet`s.
///
/// This `struct` is created by the [`difference`] method on [`LinkedSet`].
/// See its documentation for more.
///
/// [`difference`]: LinkedSet::difference
///
/// # Examples
///
/// ```
/// use toblerone::LinkedSet;
///
/// let a = LinkedSet::from([1, 2, 3]);
/// let b = LinkedSet::from([4, 2, 3, 4]);
///
/// let mut difference = a.difference(&b);
/// ```
pub struct Difference<'a, T: 'a> {
    iter: Iter<'a, T>,
    other: &'a LinkedSet<T>,
}

/// A lazy iterator producing elements in the symmetric difference of `LinkedSet`s.
///
/// This `struct` is created by the [`symmetric_difference`] method on
/// [`LinkedSet`]. See its documentation for more.
///
/// [`symmetric_difference`]: LinkedSet::symmetric_difference
///
/// # Examples
///
/// ```
/// use toblerone::LinkedSet;
///
/// let a = LinkedSet::from([1, 2, 3]);
/// let b = LinkedSet::from([4, 2, 3, 4]);
///
/// let mut intersection = a.symmetric_difference(&b);
/// ```
pub struct SymmetricDifference<'a, T: 'a> {
    iter: Chain<Difference<'a, T>, Difference<'a, T>>,
}

/// A lazy iterator producing elements in the union of `LinkedSet`s.
///
/// This `struct` is created by the [`union`] method on [`LinkedSet`].
/// See its documentation for more.
///
/// [`union`]: LinkedSet::union
///
/// # Examples
///
/// ```
/// use toblerone::LinkedSet;
///
/// let a = LinkedSet::from([1, 2, 3]);
/// let b = LinkedSet::from([4, 2, 3, 4]);
///
/// let mut union_iter = a.union(&b);
/// ```
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
            assert_eq!(head.value.as_ref(), &1);

            let tail = &*ls.tail.unwrap().as_ptr();
            assert_eq!(tail.value.as_ref(), &19);
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
            assert_eq!(head.value.as_ref(), &0);

            let tail = &*ls.tail.unwrap().as_ptr();
            assert_eq!(tail.value.as_ref(), &18);
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
    fn take() {
        let mut ls = LinkedSet::from([1, 2, 3]);
        assert_eq!(ls.take(&2), Some(2));
        assert_eq!(ls.take(&2), None);
        for (res, expected) in ls.iter().zip([1, 3].iter()) {
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn reserve() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        ls.reserve(10);
        assert!(ls.capacity() >= 10);
    }

    #[test]
    fn try_reserve() {
        let mut ls: LinkedSet<i32> = LinkedSet::new();
        ls.try_reserve(10).expect("this should'nt OOM");
        assert!(ls.capacity() >= 10);
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
