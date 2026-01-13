import asyncio
import json
from typing import Dict, Optional, Callable, Any, List

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST, extract_speaker_and_content
from .types import GroupConversationState
from prompts import prompt_loader


_PENDING_BY_TASK_KEY: Dict[str, List[Dict[str, Any]]] = {}
_LOCK_BY_TASK_KEY: Dict[str, asyncio.Lock] = {}


def _get_task_lock(task_key: str) -> asyncio.Lock:
    lock = _LOCK_BY_TASK_KEY.get(task_key)
    if lock is None:
        lock = asyncio.Lock()
        _LOCK_BY_TASK_KEY[task_key] = lock
    return lock


async def _enqueue_pending(task_key: str, item: Dict[str, Any]) -> None:
    lock = _get_task_lock(task_key)
    async with lock:
        _PENDING_BY_TASK_KEY.setdefault(task_key, []).append(item)


async def _drain_pending(task_key: str) -> List[Dict[str, Any]]:
    lock = _get_task_lock(task_key)
    async with lock:
        items = _PENDING_BY_TASK_KEY.get(task_key, [])
        if not items:
            return []
        _PENDING_BY_TASK_KEY[task_key] = []
        return items


def _build_batched_user_text(items: List[Dict[str, Any]]) -> str:
    grouped: Dict[str, List[str]] = {}
    order: List[str] = []
    for it in items:
        speaker = it.get("speaker") or ""
        text = it.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if speaker not in grouped:
            grouped[speaker] = []
            order.append(speaker)
        grouped[speaker].append(text.strip())

    if not grouped:
        return ""

    if len(order) == 1:
        only = order[0]
        lines = grouped.get(only, [])
        return "\n".join(lines).strip()

    blocks: List[str] = ["以下是多人消息，请分别对每个人回复一次："]
    for speaker in order:
        if not speaker:
            speaker_title = "(未署名)"
        else:
            speaker_title = speaker
        blocks.append(f"\n{speaker_title}:")
        for line in grouped.get(speaker, []):
            blocks.append(f"- {line}")
    return "\n".join(blocks).strip()


async def _conversation_runner(
    task_key: str,
    client_uid: str,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
) -> None:
    try:
        while True:
            batch = await _drain_pending(task_key)
            if not batch:
                return

            images = None
            if len(batch) == 1:
                images = batch[0].get("images")

            metadata: Dict[str, Any] = {"skip_history": True}

            group = chat_group_manager.get_client_group(client_uid)
            if group and len(group.members) > 1 and group.group_id == task_key:
                initiator_context = client_contexts.get(client_uid)
                default_human_name = (
                    initiator_context.character_config.human_name
                    if initiator_context
                    else "Human"
                )
                user_text = _build_batched_user_text(batch)
                if not user_text:
                    continue

                await process_group_conversation(
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_to_group,
                    group_members=group.members,
                    initiator_client_uid=client_uid,
                    user_input=user_text,
                    images=images,
                    session_emoji=np.random.choice(EMOJI_LIST),
                    metadata={
                        **metadata,
                        "default_speaker": "",
                        "human_name": default_human_name,
                    },
                )
            else:
                context = client_contexts.get(client_uid)
                if context is None:
                    return

                user_text = _build_batched_user_text(batch)
                if not user_text:
                    continue

                await process_single_conversation(
                    context=context,
                    websocket_send=websocket.send_text,
                    client_uid=client_uid,
                    user_input=user_text,
                    images=images,
                    session_emoji=np.random.choice(EMOJI_LIST),
                    metadata={
                        **metadata,
                        "default_speaker": "",
                    },
                )
    finally:
        current_conversation_tasks.pop(task_key, None)


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
) -> None:
    """Handle triggers that start a conversation"""
    metadata = None

    if msg_type == "ai-speak-signal":
        try:
            # Get proactive speak prompt from config
            prompt_name = "proactive_speak_prompt"
            prompt_file = context.system_config.tool_prompts.get(prompt_name)
            if prompt_file:
                user_input = prompt_loader.load_util(prompt_file)
            else:
                logger.warning("Proactive speak prompt not configured, using default")
                user_input = "Please say something."
        except Exception as e:
            logger.error(f"Error loading proactive speak prompt: {e}")
            user_input = "Please say something."

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "skip_memory": True,  # Skip storing in AI's internal memory
            "skip_history": True,  # Skip storing in local conversation history
        }

        await websocket.send_text(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])

    images = data.get("images")

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations
        task_key = group.group_id
        if isinstance(user_input, str) and not (metadata and metadata.get("skip_history")):
            initiator_context = client_contexts.get(client_uid)
            default_speaker = (
                initiator_context.character_config.human_name
                if initiator_context
                else "Human"
            )
            speaker, clean = extract_speaker_and_content(
                user_input, default_speaker=default_speaker
            )
            if clean and initiator_context and initiator_context.history_uid:
                for member_uid in group.members:
                    member_ctx = client_contexts.get(member_uid)
                    if member_ctx and member_ctx.history_uid:
                        store_message(
                            conf_uid=member_ctx.character_config.conf_uid,
                            history_uid=member_ctx.history_uid,
                            role="human",
                            content=clean,
                            name=speaker,
                        )
            pending_item = {"speaker": speaker, "text": clean, "images": images}
        else:
            pending_item = {"speaker": "", "text": user_input, "images": images}

        await _enqueue_pending(task_key, pending_item)

        existing = current_conversation_tasks.get(task_key)
        if existing and not existing.done():
            return

        logger.info(f"Starting queued group conversation for {task_key}")
        current_conversation_tasks[task_key] = asyncio.create_task(
            _conversation_runner(
                task_key=task_key,
                client_uid=client_uid,
                websocket=websocket,
                client_contexts=client_contexts,
                client_connections=client_connections,
                chat_group_manager=chat_group_manager,
                current_conversation_tasks=current_conversation_tasks,
                broadcast_to_group=broadcast_to_group,
            )
        )
    else:
        task_key = client_uid

        if isinstance(user_input, str) and not (metadata and metadata.get("skip_history")):
            speaker, clean = extract_speaker_and_content(
                user_input, default_speaker=context.character_config.human_name
            )
            if clean and context.history_uid:
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="human",
                    content=clean,
                    name=speaker,
                )
            pending_item = {"speaker": speaker, "text": clean, "images": images}
        else:
            pending_item = {"speaker": "", "text": user_input, "images": images}

        await _enqueue_pending(task_key, pending_item)

        existing = current_conversation_tasks.get(task_key)
        if existing and not existing.done():
            return

        logger.info(f"Starting queued conversation for {task_key}")
        current_conversation_tasks[task_key] = asyncio.create_task(
            _conversation_runner(
                task_key=task_key,
                client_uid=client_uid,
                websocket=websocket,
                client_contexts=client_contexts,
                client_connections=client_connections,
                chat_group_manager=chat_group_manager,
                current_conversation_tasks=current_conversation_tasks,
                broadcast_to_group=broadcast_to_group,
            )
        )


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
    force_cancel: bool = False,
):
    if not force_cancel:
        return

    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=heard_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
    force_cancel: bool = False,
) -> None:
    """Handles interruption for a group conversation"""
    if not force_cancel:
        return

    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
