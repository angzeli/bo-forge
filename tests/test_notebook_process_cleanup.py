"""Fault-isolated cleanup of identity-verified notebook descendants."""

import os
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from notebook_assurance.processes import OwnedProcesses


@pytest.fixture
def tracked(monkeypatch):
    owned = OwnedProcesses.__new__(OwnedProcesses)
    owned.members = {101: 1.0, 102: 2.0, 103: 3.0}
    owned.groups, owned.frozen = set(), set()
    processes = {}
    for pid, created in owned.members.items():
        process = Mock(pid=pid)
        process.create_time.return_value = created
        process.children.return_value = []
        process.status.return_value = psutil.STATUS_ZOMBIE
        processes[pid] = process
    factory = Mock(side_effect=processes.__getitem__)
    wait = Mock(side_effect=lambda live, timeout: (live, []))
    killpg = Mock()
    monkeypatch.setattr(psutil, "Process", factory)
    monkeypatch.setattr(psutil, "wait_procs", wait)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(owned, "refresh", Mock())
    monkeypatch.setattr(owned, "stop_groups", Mock())
    return SimpleNamespace(owned=owned, processes=processes, factory=factory,
                           wait=wait, killpg=killpg)


@pytest.mark.parametrize("failure", [
    PermissionError("initial inspection failed"), KeyboardInterrupt(), SystemExit(143),
])
def test_constructor_failure_cleans_descendants_before_tracker_is_assigned(
    tracked, monkeypatch, failure,
):
    tracked.processes[101].children.return_value = [
        tracked.processes[102], tracked.processes[103],
    ]
    inspections = []

    def getpgid(pid):
        inspections.append(pid)
        if len(inspections) == 1:
            tracked.processes[102].create_time.return_value = 20.0
            tracked.processes[101].children.return_value = [tracked.processes[103]]
            raise failure
        return pid
    monkeypatch.setattr(os, "getpgid", getpgid)
    owned = None

    with pytest.raises(type(failure)) as caught:
        owned = OwnedProcesses(101)

    assert owned is None
    assert caught.value is failure
    tracked.processes[101].terminate.assert_called_once_with()
    tracked.processes[103].terminate.assert_called_once_with()
    tracked.processes[102].terminate.assert_not_called()
    tracked.processes[102].kill.assert_not_called()
    assert {call.args[0] for call in tracked.killpg.call_args_list} == {101, 103}


def test_constructor_failure_preserves_cleanup_diagnostics(tracked, monkeypatch):
    original = PermissionError("initial inspection failed")
    tracked.processes[101].children.return_value = [
        tracked.processes[102], tracked.processes[103],
    ]
    monkeypatch.setattr(os, "getpgid", Mock(side_effect=[original, 102, 103, 101, 102, 103]))
    monkeypatch.setattr(OwnedProcesses, "stop_groups", Mock())
    cleanup = psutil.AccessDenied(103)
    cleanup.add_note("retained cleanup detail")
    tracked.processes[103].terminate.side_effect = cleanup

    with pytest.raises(PermissionError) as caught:
        OwnedProcesses(101)

    assert caught.value is original
    assert any("AccessDenied" in note for note in original.__notes__)
    assert "retained cleanup detail" in original.__notes__
    tracked.processes[101].terminate.assert_called_once_with()
    tracked.processes[102].terminate.assert_called_once_with()


def test_pre_freeze_inspection_failure_still_attempts_verified_descendants(tracked):
    failure = PermissionError("pre-freeze inspection failed")
    tracked.owned.refresh.side_effect = failure
    tracked.processes[102].create_time.return_value = 20.0

    with pytest.raises(PermissionError) as caught:
        tracked.owned.stop()

    assert caught.value is failure
    tracked.owned.stop_groups.assert_called_once_with()
    tracked.processes[101].terminate.assert_called_once_with()
    tracked.processes[103].terminate.assert_called_once_with()
    tracked.processes[102].terminate.assert_not_called()
    tracked.processes[102].kill.assert_not_called()
    tracked.killpg.assert_not_called()


@pytest.mark.parametrize("stage", ["lookup", "create_time", "terminate", "kill", "status"])
def test_member_access_denied_does_not_abort_other_members(tracked, stage):
    denied = psutil.AccessDenied(103)
    process = tracked.processes[103]
    if stage == "lookup":
        def lookup(pid):
            if pid == 103:
                raise denied
            return tracked.processes[pid]
        tracked.factory.side_effect = lookup
    else:
        getattr(process, stage).side_effect = denied
    if stage in {"kill", "status"}:
        tracked.wait.side_effect = lambda live, timeout: ([], live)

    with pytest.raises(psutil.AccessDenied) as caught:
        tracked.owned.stop()

    assert caught.value is denied
    for pid in (101, 102):
        tracked.processes[pid].terminate.assert_called_once_with()
        if stage in {"kill", "status"}:
            tracked.processes[pid].kill.assert_called_once_with()
    if stage in {"lookup", "create_time"}:
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
    tracked.killpg.assert_not_called()


def test_refresh_continues_after_uninspectable_child(tracked):
    denied = Mock(pid=104)
    denied.create_time.side_effect = psutil.AccessDenied(104)
    verified = Mock(pid=105)
    verified.create_time.return_value = 5.0
    tracked.processes[101].children.return_value = [denied, verified]
    tracked.processes[105] = verified

    OwnedProcesses.refresh(tracked.owned)

    assert 104 not in tracked.owned.members
    assert tracked.owned.members[105] == 5.0


def test_refresh_continues_group_inspection_after_access_denied(tracked, monkeypatch):
    def getpgid(pid):
        if pid == 101:
            raise PermissionError("group inspection denied")
        return pid
    monkeypatch.setattr(os, "getpgid", getpgid)

    with pytest.raises(PermissionError, match="group inspection denied"):
        OwnedProcesses.refresh(tracked.owned)

    assert tracked.owned.groups == {102, 103}


def test_group_uses_independently_verified_anchor(tracked, monkeypatch):
    tracked.owned.groups = {101, 999}
    tracked.processes[101].create_time.side_effect = psutil.AccessDenied(101)
    tracked.processes[103].create_time.return_value = 30.0
    monkeypatch.setattr(os, "getpgid", lambda pid: 101 if pid == 102 else 999)

    with pytest.raises(psutil.AccessDenied):
        tracked.owned.signal_groups(signal.SIGSTOP)

    tracked.killpg.assert_called_once_with(101, signal.SIGSTOP)
    assert tracked.owned.frozen == {101}


@pytest.mark.parametrize("kill_failure", [False, True])
def test_frozen_cleanup_keeps_original_error_and_attempts_remaining_members(
    tracked, monkeypatch, kill_failure,
):
    failure = PermissionError("inspection failed after freeze")
    tracked.owned.groups = {101, 102}
    tracked.owned.refresh.side_effect = [None, failure]
    monkeypatch.setattr(tracked.owned, "stop_groups",
                        OwnedProcesses.stop_groups.__get__(tracked.owned))

    def killpg(group, signum):
        if kill_failure and group == 101 and signum == signal.SIGKILL:
            raise PermissionError("frozen kill denied")
    tracked.killpg.side_effect = killpg

    with pytest.raises(PermissionError) as caught:
        tracked.owned.stop()

    assert caught.value is failure
    for group in (101, 102):
        tracked.killpg.assert_any_call(group, signal.SIGSTOP)
        tracked.killpg.assert_any_call(group, signal.SIGKILL)
    for process in tracked.processes.values():
        process.terminate.assert_called_once_with()
    if kill_failure:
        assert "Frozen-group cleanup failed: frozen kill denied" in failure.__notes__


@pytest.mark.parametrize("cancellation", [KeyboardInterrupt(), SystemExit(143)])
def test_cancellation_preserved_with_cleanup_diagnostics(tracked, cancellation):
    tracked.owned.refresh.side_effect = cancellation
    tracked.owned.stop_groups.side_effect = PermissionError("group inspection failed")
    tracked.processes[103].terminate.side_effect = psutil.AccessDenied(103)

    with pytest.raises(type(cancellation)) as caught:
        tracked.owned.stop()

    assert caught.value is cancellation
    assert any("group inspection failed" in note for note in cancellation.__notes__)
    assert any("AccessDenied" in note for note in cancellation.__notes__)
    tracked.processes[101].terminate.assert_called_once_with()
    tracked.processes[102].terminate.assert_called_once_with()


def test_wait_failure_still_kills_only_matching_identities(tracked):
    failure = psutil.AccessDenied(103)
    tracked.wait.side_effect = [failure, ([], [])]
    tracked.processes[102].create_time.side_effect = [2.0, 20.0]

    with pytest.raises(psutil.AccessDenied) as caught:
        tracked.owned.stop()

    assert caught.value is failure
    tracked.processes[101].kill.assert_called_once_with()
    tracked.processes[103].kill.assert_called_once_with()
    tracked.processes[102].kill.assert_not_called()
