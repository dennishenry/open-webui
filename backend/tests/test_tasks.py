import asyncio
import gc
import tracemalloc
from unittest.mock import AsyncMock, MagicMock

import pytest
from open_webui import tasks
from redis.exceptions import ConnectionError


def unavailable_redis():
    redis = MagicMock()
    redis.pipeline.return_value.execute = AsyncMock(side_effect=ConnectionError('redis://private-credential'))
    redis.hdel = AsyncMock(side_effect=ConnectionError('redis://private-credential'))
    redis.hget = AsyncMock(side_effect=ConnectionError('redis://private-credential'))
    return redis


@pytest.fixture(autouse=True)
def empty_registries(monkeypatch):
    monkeypatch.setattr(tasks, 'tasks', {})
    monkeypatch.setattr(tasks, 'item_tasks', {})
    monkeypatch.setattr(tasks, 'response_streams', {})
    monkeypatch.setattr(tasks, '_last_cleanup_warning', float('-inf'))
    monkeypatch.setattr(tasks, '_cleanup_failures', 0)


@pytest.mark.asyncio
@pytest.mark.parametrize('chat_id', ['chat', None, ''])
@pytest.mark.parametrize('cancelled', [False, True])
async def test_completed_tasks_release_streams_and_indexes_during_redis_outage(chat_id, cancelled, caplog):
    redis = unavailable_redis()
    redis.pipeline.return_value.execute = AsyncMock()
    task_id, task = await tasks.create_task(redis, asyncio.sleep(10 if cancelled else 0), id=chat_id)
    redis.pipeline.return_value.execute.side_effect = ConnectionError('private-credential')
    if not chat_id:
        assert chat_id not in tasks.item_tasks
        tasks.item_tasks[chat_id] = [task_id]  # Entry retained by an older worker.
    tasks.response_streams[task_id] = {'content': 'x' * 100000}
    if cancelled:
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    for _ in range(5):
        await asyncio.sleep(0)
    assert tasks.tasks == tasks.response_streams == tasks.item_tasks == {}
    assert 'OPENWEBUI_TASK_CLEANUP_FAILURE' in caplog.text
    assert 'private-credential' not in caplog.text


@pytest.mark.asyncio
async def test_stream_clear_releases_local_reference_and_bounds_diagnostics(caplog):
    redis = unavailable_redis()
    for _ in range(100):
        tasks.response_streams['task'] = {'content': 'x' * 100000}
        await tasks.clear_response_stream(redis, 'task')
    assert tasks.response_streams == {}
    assert caplog.text.count('OPENWEBUI_TASK_CLEANUP_FAILURE') == 1
    assert 'private-credential' not in caplog.text


@pytest.mark.asyncio
async def test_local_cancel_does_not_depend_on_redis():
    task_id, task = await tasks.create_task(None, asyncio.sleep(100), id='chat')
    with pytest.raises(ConnectionError):
        await tasks.stop_task(unavailable_redis(), task_id)
    await asyncio.gather(task, return_exceptions=True)
    for _ in range(5):
        await asyncio.sleep(0)
    assert task.cancelled()
    assert tasks.tasks == tasks.item_tasks == {}


@pytest.mark.asyncio
async def test_local_cleanup_precedes_blocked_redis():
    redis = unavailable_redis()
    blocked = asyncio.Event()
    redis.pipeline.return_value.execute = blocked.wait
    tasks.item_tasks[None] = ['task']
    tasks.response_streams['task'] = {'content': 'payload'}
    cleanup = asyncio.create_task(tasks.cleanup_task(redis, 'task'))
    await asyncio.sleep(0)
    assert tasks.response_streams == tasks.item_tasks == {}
    blocked.set()
    await cleanup


@pytest.mark.asyncio
async def test_fast_completion_cleans_up_after_delayed_redis_registration():
    registered = asyncio.Event()
    redis = MagicMock()
    # Use a real wait in registration so completion runs before the save finishes.
    calls = 0

    async def execute():
        nonlocal calls
        calls += 1
        if calls == 1:
            await registered.wait()

    redis.pipeline.return_value.execute = execute
    creation = asyncio.create_task(tasks.create_task(redis, asyncio.sleep(0), id='chat'))
    for _ in range(5):
        await asyncio.sleep(0)
    registered.set()
    _, task = await creation
    assert task.done()
    assert calls >= 3
    assert tasks.tasks == tasks.item_tasks == {}


@pytest.mark.asyncio
async def test_repeated_completion_and_cancellation_memory_soak():
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    for batch in range(20):
        for index in range(100):
            task_id, task = await tasks.create_task(None, asyncio.sleep(0), id='chat' if index % 2 else None)
            tasks.response_streams[task_id] = {'content': 'x' * 65536}
            if index % 3 == 0:
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for _ in range(5):
            await asyncio.sleep(0)
        assert tasks.local_task_counts()['retained_tasks'] == 0
        assert tasks.response_streams == tasks.item_tasks == {}
    gc.collect()
    retained = tracemalloc.get_traced_memory()[0] - baseline
    tracemalloc.stop()
    assert retained < 2 * 1024 * 1024
