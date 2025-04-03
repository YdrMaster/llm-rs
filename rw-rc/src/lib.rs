use std::{
    cell::Cell,
    cmp,
    hash::Hash,
    rc::{Rc, Weak},
};

/// 带有预期读写状态的引用计数。
pub struct RwRc<T> {
    /// 共享的对象和状态。
    rc: Rc<Internal<T>>,
    /// 此副本占用的读写状态。
    state: Cell<RwState>,
}

#[repr(transparent)]
pub struct RwWeak<T>(Weak<Internal<T>>);

/// 共享的对象和状态。
struct Internal<T> {
    /// 共享对象。
    val: Cell<T>,
    /// 共享读写状态。
    flag: RwFlag,
}

/// 副本读写状态。
#[derive(Clone, Copy, Debug)]
enum RwState {
    /// 持有（不关心读写）。
    Hold,
    /// 预期读，禁止修改。
    Read,
    /// 预期写，限制读写。
    Write,
}

impl<T> From<T> for RwRc<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> Clone for RwRc<T> {
    fn clone(&self) -> Self {
        Self {
            rc: self.rc.clone(),
            state: Cell::new(RwState::Hold),
        }
    }
}

impl<T> Drop for RwRc<T> {
    fn drop(&mut self) {
        self.release()
    }
}

impl<T> RwRc<T> {
    pub fn new(val: T) -> Self {
        Self {
            rc: Rc::new(Internal {
                val: Cell::new(val),
                flag: RwFlag::new(),
            }),
            state: Cell::new(RwState::Hold),
        }
    }

    pub fn try_read(&self) -> Option<&T> {
        match self.state.get() {
            RwState::Hold => {
                if !self.rc.flag.read() {
                    return None;
                }
            }
            RwState::Read | RwState::Write => {}
        }
        self.state.set(RwState::Read);
        Some(unsafe { &*self.rc.val.as_ptr() })
    }

    pub fn try_write(&self) -> Option<&mut T> {
        match self.state.get() {
            RwState::Hold => {
                if !self.rc.flag.write() {
                    return None;
                }
            }
            RwState::Read => {
                self.rc.flag.release_read();
                assert!(self.rc.flag.write())
            }
            RwState::Write => {}
        }
        self.state.set(RwState::Write);
        Some(unsafe { &mut *self.rc.val.as_ptr() })
    }

    pub fn release(&self) {
        match self.state.replace(RwState::Hold) {
            RwState::Hold => {}
            RwState::Read => self.rc.flag.release_read(),
            RwState::Write => self.rc.flag.release_write(),
        }
    }

    pub fn read(&self) -> &T {
        self.try_read().unwrap()
    }

    pub fn write(&self) -> &mut T {
        self.try_write().unwrap()
    }

    pub fn weak(&self) -> RwWeak<T> {
        RwWeak(Rc::downgrade(&self.rc))
    }
}

impl<T> Clone for RwWeak<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> PartialEq for RwWeak<T> {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for RwWeak<T> {}

impl<T> Hash for RwWeak<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

impl<T> PartialOrd for RwWeak<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for RwWeak<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&self.0.as_ptr(), &other.0.as_ptr())
    }
}

impl<T> RwWeak<T> {
    pub fn hold(&self) -> Option<RwRc<T>> {
        self.0.upgrade().map(|rc| RwRc {
            rc,
            state: Cell::new(RwState::Hold),
        })
    }
}

/// 共享读写状态。
#[repr(transparent)]
struct RwFlag(Cell<usize>);

impl RwFlag {
    /// 初始化状态变量。
    fn new() -> Self {
        Self(Cell::new(0))
    }

    /// 尝试进入读状态，失败时状态不变。
    fn read(&self) -> bool {
        match self.0.get() {
            usize::MAX => false,
            n => {
                self.0.set(n + 1);
                true
            }
        }
    }

    /// 尝试进入写状态，失败时状态不变。
    fn write(&self) -> bool {
        match self.0.get() {
            0 => {
                self.0.set(usize::MAX);
                true
            }
            _ => false,
        }
    }

    /// 释放读状态，当前必须在读状态。
    fn release_read(&self) {
        let current = self.0.get();
        debug_assert!((1..usize::MAX).contains(&current));
        self.0.set(current - 1)
    }

    /// 释放写状态，当前必须在写状态。
    fn release_write(&self) {
        let current = self.0.get();
        debug_assert_eq!(current, usize::MAX);
        self.0.set(0)
    }
}
