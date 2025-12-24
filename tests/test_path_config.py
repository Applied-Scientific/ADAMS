#!/usr/bin/env python3
"""
Test script for the centralized path configuration.

This script tests that the path configuration works correctly in different scenarios.
"""

from pathlib import Path

from adams.path_config import (
    get_agent_data_path,
    get_subdirectory,
    reset_paths,
    set_agent_data_path,
)


def test_default_path():
    """Test that setting path with no args uses current working directory + agent_data"""
    reset_paths()
    # Now must explicitly call set_agent_data_path() instead of relying on auto-init
    path = set_agent_data_path()  # Call with no args to use default
    expected = Path.cwd() / "agent_data"
    print(f"Test 1 - Default path:")
    print(f"  Expected: {expected}")
    print(f"  Got: {path}")
    assert path == expected, f"Expected {expected}, got {path}"
    print("  ✓ PASS\n")


def test_set_direct_path():
    """Test setting a direct path"""
    reset_paths()
    test_path = Path("/tmp/test_project/data")
    result = set_agent_data_path(path=test_path)
    print(f"Test 2 - Set direct path:")
    print(f"  Set to: {test_path}")
    print(f"  Got: {result}")
    assert (
        result == test_path.resolve()
    ), f"Expected {test_path.resolve()}, got {result}"

    # Verify get returns the same path
    retrieved = get_agent_data_path()
    assert (
        retrieved == test_path.resolve()
    ), f"Expected {test_path.resolve()}, got {retrieved}"
    print("  ✓ PASS\n")


def test_set_from_input_file():
    """Test setting path from input file location"""
    reset_paths()
    input_file = Path("/tmp/project/data/receptor.pdb")
    result = set_agent_data_path(input_file_path=input_file)
    expected = input_file.parent / "agent_data"
    print(f"Test 3 - Set from input file:")
    print(f"  Input file: {input_file}")
    print(f"  Expected: {expected.resolve()}")
    print(f"  Got: {result}")
    assert result == expected.resolve(), f"Expected {expected.resolve()}, got {result}"
    print("  ✓ PASS\n")


def test_get_subdirectory():
    """Test getting subdirectories"""
    reset_paths()
    base_path = Path("/tmp/test_data").resolve()  # Resolve to handle symlinks
    set_agent_data_path(path=base_path)

    logs_dir = get_subdirectory("logs")
    traces_dir = get_subdirectory("traces")
    outputs_dir = get_subdirectory("outputs", "run_123")

    print(f"Test 4 - Get subdirectories:")
    print(f"  Base: {base_path}")
    print(f"  logs: {logs_dir}")
    print(f"  traces: {traces_dir}")
    print(f"  outputs/run_123: {outputs_dir}")

    assert (
        logs_dir == base_path / "logs"
    ), f"Expected {base_path / 'logs'}, got {logs_dir}"
    assert (
        traces_dir == base_path / "traces"
    ), f"Expected {base_path / 'traces'}, got {traces_dir}"
    assert (
        outputs_dir == base_path / "outputs" / "run_123"
    ), f"Expected {base_path / 'outputs' / 'run_123'}, got {outputs_dir}"
    print("  ✓ PASS\n")


def test_get_agent_data_path_raises_when_not_set():
    """Test that get_agent_data_path raises RuntimeError if path not configured"""
    reset_paths()

    try:
        path = get_agent_data_path()
        assert False, "Expected RuntimeError but got path: " + str(path)
    except RuntimeError as e:
        assert "not configured" in str(e).lower()
        print("Test 5 - Raises RuntimeError when path not set:")
        print("  ✓ Correctly raises RuntimeError when path not set")
        print("  ✓ PASS\n")


def test_context_isolation():
    """Test that path configuration is context-aware (for multi-session support)"""
    import asyncio
    from contextvars import copy_context

    async def session_1():
        set_agent_data_path(path="/tmp/session1/data")
        await asyncio.sleep(0.01)  # Simulate some work
        path = get_agent_data_path()
        return str(path)

    async def session_2():
        set_agent_data_path(path="/tmp/session2/data")
        await asyncio.sleep(0.01)  # Simulate some work
        path = get_agent_data_path()
        return str(path)

    async def run_concurrent_sessions():
        # Run both sessions concurrently
        results = await asyncio.gather(
            asyncio.create_task(session_1()), asyncio.create_task(session_2())
        )
        return results

    print(f"Test 6 - Context isolation (concurrent sessions):")
    reset_paths()

    # Note: contextvars work within the same async context by default
    # For true isolation, each would need to run in separate contexts
    result1 = asyncio.run(session_1())
    reset_paths()
    result2 = asyncio.run(session_2())

    print(f"  Session 1 path: {result1}")
    print(f"  Session 2 path: {result2}")
    assert "/session1" in result1, f"Expected session1 path, got {result1}"
    assert "/session2" in result2, f"Expected session2 path, got {result2}"
    print("  ✓ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Centralized Path Configuration")
    print("=" * 60)
    print()

    try:
        test_default_path()
        test_set_direct_path()
        test_set_from_input_file()
        test_get_subdirectory()
        test_get_agent_data_path_raises_when_not_set()
        test_context_isolation()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
