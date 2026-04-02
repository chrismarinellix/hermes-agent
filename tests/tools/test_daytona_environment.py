"""Unit tests for the Daytona cloud sandbox environment backend."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock Daytona SDK objects
# ---------------------------------------------------------------------------

def _make_exec_response(result="", exit_code=0):
    return SimpleNamespace(result=result, exit_code=exit_code)


def _make_sandbox(sandbox_id="sb-123", state="started"):
    sb = MagicMock()
    sb.id = sandbox_id
    sb.state = state
    sb.process.exec.return_value = _make_exec_response()
    return sb


def _patch_daytona_imports(monkeypatch):
    """Patch the daytona SDK so DaytonaEnvironment can be imported without it."""
    import types as _types

    import enum

    class _SandboxState(str, enum.Enum):
        STARTED = "started"
        STOPPED = "stopped"
        ARCHIVED = "archived"
        ERROR = "error"

    daytona_mod = _types.ModuleType("daytona")
    daytona_mod.Daytona = MagicMock
    daytona_mod.CreateSandboxFromImageParams = MagicMock
    daytona_mod.DaytonaError = type("DaytonaError", (Exception,), {})
    daytona_mod.Resources = MagicMock(name="Resources")
    daytona_mod.SandboxState = _SandboxState

    monkeypatch.setitem(__import__("sys").modules, "daytona", daytona_mod)
    return daytona_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def daytona_sdk(monkeypatch):
    """Provide a mock daytona SDK module and return it for assertions."""
    return _patch_daytona_imports(monkeypatch)


@pytest.fixture()
def make_env(daytona_sdk, monkeypatch):
    """Factory that creates a DaytonaEnvironment with a mocked SDK."""
    # Prevent is_interrupted from interfering
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    # Prevent skills/credential sync from consuming mock exec calls
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])

    def _factory(
        sandbox=None,
        get_side_effect=None,
        list_return=None,
        home_dir="/root",
        persistent=True,
        **kwargs,
    ):
        sandbox = sandbox or _make_sandbox()
        # Mock the $HOME detection
        sandbox.process.exec.return_value = _make_exec_response(result=home_dir)

        mock_client = MagicMock()
        mock_client.create.return_value = sandbox

        if get_side_effect is not None:
            mock_client.get.side_effect = get_side_effect
        else:
            # Default: no existing sandbox found via get()
            mock_client.get.side_effect = daytona_sdk.DaytonaError("not found")

        # Default: no legacy sandbox found via list()
        if list_return is not None:
            mock_client.list.return_value = list_return
        else:
            mock_client.list.return_value = SimpleNamespace(items=[])

        daytona_sdk.Daytona = MagicMock(return_value=mock_client)

        from tools.environments.daytona import DaytonaEnvironment

        kwargs.setdefault("disk", 10240)
        env = DaytonaEnvironment(
            image="test-image:latest",
            persistent_filesystem=persistent,
            **kwargs,
        )
        env._mock_client = mock_client  # expose for assertions
        return env

    return _factory


# ---------------------------------------------------------------------------
# Constructor / cwd resolution
# ---------------------------------------------------------------------------

class TestCwdResolution:
    def test_default_cwd_resolves_home(self, make_env):
        env = make_env(home_dir="/home/testuser")
        assert env.cwd == "/home/testuser"

    def test_tilde_cwd_resolves_home(self, make_env):
        env = make_env(cwd="~", home_dir="/home/testuser")
        assert env.cwd == "/home/testuser"

    def test_explicit_cwd_not_overridden(self, make_env):
        """Explicit cwd should be set before init_session.

        After init_session(), the cwdfile may update cwd to whatever the
        login shell reports.  We make the mock return /workspace for the
        cwdfile read so init_session doesn't override the explicit cwd.
        """
        sb = _make_sandbox()
        # Return /workspace for all exec calls including init_session's
        # snapshot bootstrap and cwdfile reads
        sb.process.exec.return_value = _make_exec_response(result="/workspace")
        env = make_env(sandbox=sb, cwd="/workspace", home_dir="/workspace")
        assert env.cwd == "/workspace"

    def test_home_detection_failure_keeps_default_cwd(self, make_env):
        """When $HOME detection fails, cwd falls back to constructor default.

        init_session() still runs but its cwdfile read returns empty,
        so cwd is not overwritten.
        """
        sb = _make_sandbox()
        call_count = {"n": 0}

        def _exec_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # $HOME detection fails
                raise RuntimeError("exec failed")
            # All subsequent calls (init_session, cwdfile reads) succeed
            # but return empty so they don't override cwd
            return _make_exec_response(result="", exit_code=0)

        sb.process.exec.side_effect = _exec_side_effect
        env = make_env(sandbox=sb)
        assert env.cwd == "/home/daytona"

    def test_empty_home_keeps_default_cwd(self, make_env):
        env = make_env(home_dir="")
        assert env.cwd == "/home/daytona"  # keeps constructor default


# ---------------------------------------------------------------------------
# Sandbox persistence / resume
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistent_resumes_via_get(self, make_env):
        existing = _make_sandbox(sandbox_id="sb-existing")
        existing.process.exec.return_value = _make_exec_response(result="/root")
        env = make_env(get_side_effect=lambda name: existing, persistent=True,
                       task_id="mytask")
        existing.start.assert_called_once()
        env._mock_client.get.assert_called_once_with("hermes-mytask")
        env._mock_client.create.assert_not_called()

    def test_persistent_resumes_legacy_via_list(self, make_env, daytona_sdk):
        legacy = _make_sandbox(sandbox_id="sb-legacy")
        legacy.process.exec.return_value = _make_exec_response(result="/root")
        env = make_env(
            get_side_effect=daytona_sdk.DaytonaError("not found"),
            list_return=SimpleNamespace(items=[legacy]),
            persistent=True,
            task_id="mytask",
        )
        legacy.start.assert_called_once()
        env._mock_client.list.assert_called_once_with(
            labels={"hermes_task_id": "mytask"}, page=1, limit=1)
        env._mock_client.create.assert_not_called()

    def test_persistent_creates_new_when_none_found(self, make_env, daytona_sdk):
        env = make_env(
            get_side_effect=daytona_sdk.DaytonaError("not found"),
            persistent=True,
            task_id="mytask",
        )
        env._mock_client.create.assert_called_once()
        # Verify the name and labels were passed to CreateSandboxFromImageParams
        # by checking get() was called with the right sandbox name
        env._mock_client.get.assert_called_with("hermes-mytask")
        env._mock_client.list.assert_called_with(
            labels={"hermes_task_id": "mytask"}, page=1, limit=1)

    def test_non_persistent_skips_lookup(self, make_env):
        env = make_env(persistent=False)
        env._mock_client.get.assert_not_called()
        env._mock_client.list.assert_not_called()
        env._mock_client.create.assert_called_once()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_persistent_cleanup_stops_sandbox(self, make_env):
        env = make_env(persistent=True)
        sb = env._sandbox
        env.cleanup()
        sb.stop.assert_called_once()

    def test_non_persistent_cleanup_deletes_sandbox(self, make_env):
        env = make_env(persistent=False)
        sb = env._sandbox
        env.cleanup()
        env._mock_client.delete.assert_called_once_with(sb)

    def test_cleanup_idempotent(self, make_env):
        env = make_env(persistent=True)
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_swallows_errors(self, make_env):
        env = make_env(persistent=True)
        env._sandbox.stop.side_effect = RuntimeError("stop failed")
        env.cleanup()  # should not raise
        assert env._sandbox is None


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_basic_command(self, make_env):
        sb = _make_sandbox()
        # Calls: $HOME detection, init_session bootstrap, init_session cat,
        # _before_execute sandbox refresh, _run_bash command, _update_cwd cat
        sb.process.exec.return_value = _make_exec_response(result="/root")
        sb.state = "started"
        env = make_env(sandbox=sb)

        # Reset mock to control just the execute() calls
        sb.process.exec.reset_mock()
        sb.process.exec.return_value = _make_exec_response(result="hello", exit_code=0)
        result = env.execute("echo hello")
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_command_wrapped_with_shell_timeout(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.return_value = _make_exec_response(result="/root")
        sb.state = "started"
        env = make_env(sandbox=sb, timeout=42)

        sb.process.exec.reset_mock()
        sb.process.exec.return_value = _make_exec_response(result="ok", exit_code=0)
        env.execute("echo hello")
        # The command sent to _DaytonaProcessHandle should be wrapped with
        # `timeout N bash -c '...'`
        call_args = sb.process.exec.call_args_list[-1]
        cmd = call_args[0][0]
        assert "timeout 42 bash -c " in cmd

    def test_timeout_returns_exit_code_124(self, make_env):
        """Shell timeout utility returns exit code 124."""
        sb = _make_sandbox()
        sb.process.exec.return_value = _make_exec_response(result="/root")
        sb.state = "started"
        env = make_env(sandbox=sb)

        sb.process.exec.reset_mock()
        sb.process.exec.return_value = _make_exec_response(result="", exit_code=124)
        result = env.execute("sleep 300", timeout=5)
        assert result["returncode"] == 124

    def test_nonzero_exit_code(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.return_value = _make_exec_response(result="/root")
        sb.state = "started"
        env = make_env(sandbox=sb)

        sb.process.exec.reset_mock()
        sb.process.exec.return_value = _make_exec_response(result="not found", exit_code=127)
        result = env.execute("bad_cmd")
        assert result["returncode"] == 127

    def test_stdin_data_wraps_heredoc(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.return_value = _make_exec_response(result="/root")
        sb.state = "started"
        env = make_env(sandbox=sb)

        sb.process.exec.reset_mock()
        sb.process.exec.return_value = _make_exec_response(result="ok", exit_code=0)
        env.execute("python3", stdin_data="print('hi')")
        # Check that one of the exec calls contains heredoc markers.
        # The last call may be the cwdfile read, so check all calls.
        all_cmds = [
            call_args[0][0]
            for call_args in sb.process.exec.call_args_list
        ]
        heredoc_cmd = [c for c in all_cmds if "HERMES_EOF_" in c]
        assert heredoc_cmd, f"No heredoc found in exec calls: {all_cmds}"
        cmd = heredoc_cmd[0]
        assert "print" in cmd
        assert "hi" in cmd

    def test_custom_cwd_passed_through(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.return_value = _make_exec_response(result="/root")
        sb.state = "started"
        env = make_env(sandbox=sb)

        sb.process.exec.reset_mock()
        sb.process.exec.return_value = _make_exec_response(result="/tmp", exit_code=0)
        env.execute("pwd", cwd="/tmp")
        # In the unified model, cwd is embedded in the _wrap_command output
        # and the _DaytonaProcessHandle also passes cwd to the SDK
        call_args = sb.process.exec.call_args_list[-1]
        cmd = call_args[0][0]
        # The wrapped command includes a cd to the cwd
        assert "/tmp" in cmd

    def test_daytona_error_returns_error_result(self, make_env, daytona_sdk):
        """In the unified model, SDK errors are caught by _DaytonaProcessHandle
        and returned as error results (no automatic retry)."""
        sb = _make_sandbox()
        sb.state = "started"
        sb.process.exec.return_value = _make_exec_response(result="/root")
        env = make_env(sandbox=sb)

        sb.process.exec.reset_mock()
        sb.process.exec.side_effect = daytona_sdk.DaytonaError("transient")
        result = env.execute("echo retry")
        assert result["returncode"] == 1
        assert "Daytona execution error" in result["output"]


# ---------------------------------------------------------------------------
# Resource conversion
# ---------------------------------------------------------------------------

class TestResourceConversion:
    def _get_resources_kwargs(self, daytona_sdk):
        return daytona_sdk.Resources.call_args.kwargs

    def test_memory_converted_to_gib(self, make_env, daytona_sdk):
        env = make_env(memory=5120)
        assert self._get_resources_kwargs(daytona_sdk)["memory"] == 5

    def test_disk_converted_to_gib(self, make_env, daytona_sdk):
        env = make_env(disk=10240)
        assert self._get_resources_kwargs(daytona_sdk)["disk"] == 10

    def test_small_values_clamped_to_1(self, make_env, daytona_sdk):
        env = make_env(memory=100, disk=100)
        kw = self._get_resources_kwargs(daytona_sdk)
        assert kw["memory"] == 1
        assert kw["disk"] == 1


# ---------------------------------------------------------------------------
# Ensure sandbox ready
# ---------------------------------------------------------------------------

class TestInterrupt:
    def test_interrupt_returns_130(self, make_env, monkeypatch):
        """In the unified model, interrupt is handled by BaseEnvironment._wait_for_process."""
        sb = _make_sandbox()
        sb.state = "started"
        sb.process.exec.return_value = _make_exec_response(result="/root")
        env = make_env(sandbox=sb)

        # Make the SDK exec block long enough for the interrupt check to fire
        import time as time_mod
        def slow_exec(*args, **kwargs):
            time_mod.sleep(5)
            return _make_exec_response(result="done", exit_code=0)

        sb.process.exec.reset_mock()
        sb.process.exec.side_effect = slow_exec

        # Patch is_interrupted in the base module where _wait_for_process uses it
        monkeypatch.setattr(
            "tools.environments.base.is_interrupted", lambda: True
        )
        result = env.execute("sleep 10")
        assert result["returncode"] == 130


# ---------------------------------------------------------------------------
# SDK error handling
# ---------------------------------------------------------------------------

class TestSdkError:
    def test_sdk_error_returns_error_result(self, make_env, daytona_sdk):
        """SDK errors in _DaytonaProcessHandle are caught and returned cleanly."""
        sb = _make_sandbox()
        sb.state = "started"
        sb.process.exec.return_value = _make_exec_response(result="/root")
        env = make_env(sandbox=sb)

        sb.process.exec.reset_mock()
        sb.process.exec.side_effect = daytona_sdk.DaytonaError("fail")
        result = env.execute("echo x")
        assert result["returncode"] == 1
        assert "Daytona execution error" in result["output"]


# ---------------------------------------------------------------------------
# Ensure sandbox ready
# ---------------------------------------------------------------------------

class TestEnsureSandboxReady:
    def test_restarts_stopped_sandbox(self, make_env):
        env = make_env()
        env._sandbox.state = "stopped"
        env._ensure_sandbox_ready()
        env._sandbox.start.assert_called()

    def test_no_restart_when_running(self, make_env):
        env = make_env()
        env._sandbox.state = "started"
        env._ensure_sandbox_ready()
        env._sandbox.start.assert_not_called()
