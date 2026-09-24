"""Track owned descendants by PID and creation time, including kernel sessions."""

import os
import signal
import sys

import psutil  # Installed by the existing ipykernel development dependency.


def _raise_cleanup_errors(errors):
    if errors:
        original = next((exc for exc in errors if not isinstance(exc, Exception)), errors[0])
        for exc in errors:
            if exc is not original:
                original.add_note(f"{type(exc).__name__}: {exc}")
                for note in getattr(exc, "__notes__", ()):
                    original.add_note(note)
        raise original


def _wait_for_members(members, errors):
    try:
        return psutil.wait_procs(members, timeout=3)[1]
    except BaseException as exc:
        # Cleanup boundary: retain the diagnostic and still attempt verified kills.
        errors.append(exc)
        return members


class OwnedProcesses:
    def __init__(self, pid):
        self.members = {pid: psutil.Process(pid).create_time()}
        self.groups = set()
        self.frozen = set()
        try:
            self.refresh()
        except BaseException as original:
            # Ownership boundary: the caller cannot retain a failed constructor's tracker.
            try:
                self.stop()
            except BaseException as cleanup:
                # Keep cleanup diagnostics without replacing the construction failure.
                if cleanup is not original:
                    original.add_note(
                        f"Constructor cleanup failed: {type(cleanup).__name__}: {cleanup}")
                    for note in getattr(cleanup, "__notes__", ()):
                        original.add_note(note)
            raise

    def refresh(self):
        for pid, created in list(self.members.items()):
            try:
                process = psutil.Process(pid)
                if process.create_time() == created:
                    for child in process.children(recursive=True):
                        try:
                            self.members[child.pid] = child.create_time()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if os.name == "posix":
            errors = []
            for pid, created in list(self.members.items()):
                try:
                    if psutil.Process(pid).create_time() == created and os.getpgid(pid) == pid:
                        self.groups.add(pid)
                except (psutil.NoSuchProcess, ProcessLookupError):
                    continue
                except (psutil.AccessDenied, OSError) as exc:
                    errors.append(exc)
            _raise_cleanup_errors(errors)

    def signal_groups(self, signum):
        errors = []
        for group in self.groups:
            # A group is eligible only while a recorded member still anchors its ownership.
            for pid, created in self.members.items():
                try:
                    if (psutil.Process(pid).create_time() == created
                            and os.getpgid(pid) == group):
                        os.killpg(group, signum)
                        if signum == signal.SIGSTOP:
                            self.frozen.add(group)
                        break
                except (psutil.NoSuchProcess, ProcessLookupError):
                    continue
                except (psutil.AccessDenied, OSError) as exc:
                    errors.append(exc)
        _raise_cleanup_errors(errors)

    def stop_groups(self):
        # Freeze owned worker/kernel groups before the final descendant snapshot.
        try:
            self.signal_groups(signal.SIGSTOP)
            self.refresh()
            self.signal_groups(signal.SIGSTOP)
            self.refresh()
        finally:
            # Cleanup boundary: even a failed inspection must not leave frozen kernels.
            # Frozen group leaders cannot exit or recycle their verified group IDs.
            original = sys.exception()
            errors = []
            for group in self.frozen:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    errors.append(str(exc))
            if errors:
                message = "Frozen-group cleanup failed: " + "; ".join(errors)
                if original is not None:
                    original.add_note(message)
                else:
                    raise RuntimeError(message)

    def _signal_members(self, pids, action, errors):
        live = []
        for pid in pids:
            try:
                process = psutil.Process(pid)
                if process.create_time() == self.members[pid]:
                    live.append(process)
                    getattr(process, action)()
            except psutil.NoSuchProcess:
                continue
            except BaseException as exc:
                # Cleanup boundary: defer this failure until other verified members are signaled.
                errors.append(exc)
        return live

    def stop(self):
        errors = []
        actions = [self.refresh]
        if os.name == "posix":
            actions.append(self.stop_groups)
        for action in actions:
            try:
                action()
            except BaseException as exc:
                # Cleanup boundary: inspection failures must not strand other owned members.
                errors.append(exc)
        live = self._signal_members(reversed(list(self.members)), "terminate", errors)
        remaining = _wait_for_members(live, errors)
        self._signal_members((process.pid for process in remaining), "kill", errors)
        survivors = _wait_for_members(remaining, errors)
        for process in survivors:
            try:
                if (psutil.Process(process.pid).create_time() == self.members[process.pid]
                        and process.status() != psutil.STATUS_ZOMBIE):
                    errors.append(RuntimeError("Owned notebook descendants survived termination."))
            except psutil.NoSuchProcess:
                pass
            except BaseException as exc:
                # Cleanup boundary: retain every survivor-inspection diagnostic before raising.
                errors.append(exc)
        _raise_cleanup_errors(errors)
