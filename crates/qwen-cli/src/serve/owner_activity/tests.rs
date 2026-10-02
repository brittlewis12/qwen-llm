use super::*;
use std::sync::mpsc::channel;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(2);

#[test]
fn preparation_and_pending_completions_both_preclude_idle() {
    let mut owner = OwnerActivity::default();
    let guard = owner.admission().try_admit().unwrap();
    owner.idle_if_quiet(|| panic!("preparation is active even without queued execution"));
    drop(guard);
    owner.idle_if_quiet(|| panic!("completion has not been delivered"));
    let mut calls = 0;
    owner.drain_finished(|| calls += 1);
    owner.drain_finished(|| panic!("completion delivered twice"));
    assert_eq!(calls, 1);
    owner.idle_if_quiet(|| calls += 1);
    assert_eq!(calls, 2);
    assert!(owner.is_settled());
}

#[test]
fn closing_keeps_outstanding_completions_and_refuses_new_admission() {
    let mut owner = OwnerActivity::default();
    let admission = owner.admission();
    let guard = admission.try_admit().unwrap();
    admission.close();
    assert!(admission.try_admit().is_none());
    assert!(!owner.is_settled());
    drop(guard);
    let mut calls = 0;
    owner.drain_finished(|| calls += 1);
    assert_eq!(calls, 1);
    assert!(owner.is_settled());
    owner.idle_if_quiet(|| panic!("closed owners do not publish idle work"));
}

#[test]
fn read_only_release_does_not_complete_or_release_twice() {
    let mut owner = OwnerActivity::default();
    let admission = owner.admission();
    let ordinary = admission.try_admit().unwrap();
    admission.try_admit().unwrap().release_read_only();
    owner.drain_finished(|| panic!("read-only activity is not a request completion"));
    owner.idle_if_quiet(|| panic!("ordinary activity remains active"));
    drop(ordinary);
    let mut calls = 0;
    owner.drain_finished(|| calls += 1);
    assert_eq!(calls, 1);
    assert!(owner.is_settled());
}

#[test]
fn completion_during_drain_prevents_idle_and_callbacks_stay_on_owner() {
    let mut owner = OwnerActivity::default();
    let admission = owner.admission();
    drop(admission.try_admit().unwrap());
    let guard = admission.try_admit().unwrap();
    let (release, released) = channel();
    let (done, completed) = channel();
    let worker = std::thread::spawn(move || {
        released.recv_timeout(WAIT).unwrap();
        drop(guard);
        done.send(()).unwrap();
    });
    let owner_thread = std::thread::current().id();
    owner.drain_finished(|| {
        assert_eq!(std::thread::current().id(), owner_thread);
        release.send(()).unwrap();
        completed.recv_timeout(WAIT).unwrap();
    });
    owner.idle_if_quiet(|| panic!("a new completion arrived during the drain"));
    let mut calls = 0;
    owner.drain_finished(|| {
        assert_eq!(std::thread::current().id(), owner_thread);
        calls += 1;
    });
    assert_eq!(calls, 1);
    assert!(owner.is_settled());
    worker.join().unwrap();
}

#[test]
fn maintenance_holds_admission_boundary_until_callback_returns() {
    let mut owner = OwnerActivity::default();
    let admission = owner.admission();
    let (start, started) = channel();
    let (checked, checks) = channel();
    let worker = std::thread::spawn(move || {
        started.recv_timeout(WAIT).unwrap();
        // Deterministically observe the boundary, rather than infer it from a
        // thread failing to make progress within an arbitrary sleep interval.
        assert!(matches!(
            admission.0.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        ));
        checked.send(()).unwrap();
        admission.try_admit().unwrap()
    });
    owner.idle_if_quiet(|| {
        start.send(()).unwrap();
        checks.recv_timeout(WAIT).unwrap();
    });
    let guard = worker.join().unwrap();
    owner.idle_if_quiet(|| panic!("admission after maintenance is now active"));
    drop(guard);
    owner.drain_finished(|| {});
    assert!(owner.is_settled());
}

#[test]
fn poisoned_accounting_fails_closed_and_drop_does_not_panic() {
    let mut owner = OwnerActivity::default();
    let admission = owner.admission();
    let guard = admission.try_admit().unwrap();
    let poisoned = admission.clone();
    assert!(
        std::thread::spawn(move || {
            let _lock = poisoned.0.lock().unwrap();
            panic!("poison accounting");
        })
        .join()
        .is_err()
    );
    assert!(admission.try_admit().is_none());
    let unwind = std::panic::catch_unwind(move || {
        let _guard = guard;
        panic!("unwind outstanding request");
    });
    assert!(unwind.is_err());
    admission.close();
    let mut calls = 0;
    owner.drain_finished(|| calls += 1);
    assert_eq!(calls, 1);
    owner.idle_if_quiet(|| panic!("poison must not become idle"));
    assert!(!owner.is_settled());
}
