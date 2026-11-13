//! Host QoS for threads that feed the GPU.
//!
//! A blocking wait for a command buffer can lose several milliseconds while the
//! host CPU is saturated: the GPU finishes, but the waiter is not scheduled to
//! observe it. Raising the calling thread to user-initiated for the duration of
//! a decode, prefill, scheduler round or request burst removes that idle. The
//! guard is nested-safe and never lowers a thread already at or above
//! user-initiated.
crate::env_flag!(
    /// Raise the thread that feeds the GPU to user-initiated QoS for the duration of a decode, prefill, scheduler round or request/packet burst, so it is scheduled promptly when the host CPU is saturated. 0 disables every raise.
    default_on qos_enabled, "QWEN_GPU_THREAD_QOS"
);

const QOS_CLASS_USER_INITIATED: u32 = 0x19;
const QOS_CLASS_DEFAULT: u32 = 0x15;

unsafe extern "C" {
    fn pthread_get_qos_class_np(
        thread: libc::pthread_t,
        qos_class: *mut u32,
        relative_priority: *mut i32,
    ) -> i32;
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

/// Restores the previous QoS class when dropped.
pub struct GpuThreadQos {
    previous: Option<(u32, i32)>,
}

impl GpuThreadQos {
    pub fn enter() -> Self {
        Self::enter_if(qos_enabled())
    }

    /// Raise unless `on` is false or the thread is already at user-initiated or
    /// higher. Safe to nest: an inner guard sees the raised class and does
    /// nothing, so only the outer guard restores.
    pub fn enter_if(on: bool) -> Self {
        if !on {
            return Self { previous: None };
        }
        let (mut class, mut relative) = (QOS_CLASS_DEFAULT, 0);
        // SAFETY: both out-pointers are valid for the call.
        let rc =
            unsafe { pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative) };
        if rc != 0 {
            // Unknown class: raise and restore the default on drop.
            // SAFETY: no preconditions.
            unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) };
            return Self {
                previous: Some((QOS_CLASS_DEFAULT, 0)),
            };
        }
        if class >= QOS_CLASS_USER_INITIATED {
            return Self { previous: None };
        }
        // SAFETY: no preconditions.
        unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) };
        Self {
            previous: Some((class, relative)),
        }
    }
}

impl Drop for GpuThreadQos {
    fn drop(&mut self) {
        if let Some((class, relative)) = self.previous {
            // SAFETY: no preconditions.
            unsafe { pthread_set_qos_class_self_np(class, relative) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn current() -> (u32, i32) {
        let (mut class, mut relative) = (0, 0);
        assert_eq!(
            unsafe { pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative) },
            0
        );
        (class, relative)
    }
    #[test]
    fn qos_scope_nesting_and_early_return_restore_the_feeding_thread() {
        let before = current();
        let early = || -> Result<(), ()> {
            let _slice = GpuThreadQos::enter_if(true);
            assert!(current().0 >= QOS_CLASS_USER_INITIATED);
            {
                let nested = GpuThreadQos::enter_if(true);
                assert!(nested.previous.is_none());
            }
            Err(())
        };
        assert!(early().is_err());
        assert_eq!(current(), before);
        let disabled = GpuThreadQos::enter_if(false);
        assert!(disabled.previous.is_none());
        assert_eq!(current(), before);
    }
}
