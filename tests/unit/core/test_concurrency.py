import threading

from app.core.concurrency import AdmissionGate


def test_unlimited_gate_always_admits():
    g = AdmissionGate(0)
    assert all(g.try_enter() for _ in range(100)) and g.in_flight == 100


def test_gate_refuses_beyond_limit_and_recovers_after_leave():
    g = AdmissionGate(2)
    assert g.try_enter() and g.try_enter()
    assert not g.try_enter()
    g.leave()
    assert g.try_enter() and g.in_flight == 2


def test_leave_never_goes_negative():
    g = AdmissionGate(1)
    g.leave()
    assert g.in_flight == 0


def test_gate_is_thread_safe():
    g = AdmissionGate(5)
    admitted = []

    def grab():
        admitted.append(g.try_enter())

    threads = [threading.Thread(target=grab) for _ in range(50)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(admitted) == 5 and g.in_flight == 5
