import asyncio
from typing import Sequence, List, Optional, Any, Union, Mapping
from autogen_core import FunctionCall
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    RequestUsage,
    ModelCapabilities,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionExecutionResultMessage
)
from autogen_core.tools import Tool
from ollama import AsyncClient, ResponseError
import json

from app.core.config import (
    OLLAMA_REQUEST_TIMEOUT_SEC,
    OLLAMA_MAX_RETRIES,
    OLLAMA_RETRY_BASE_DELAY_SEC,
    ollama_async_client_kwargs,
)


class SimpleOllamaClient(ChatCompletionClient):
    """
    A simple wrapper around the official Ollama Python SDK for AutoGen 0.4.
    Supports tool calling for code execution.
    """
    def __init__(self, model: str, host: str, **kwargs):
        self._model = model
        self._client = AsyncClient(**ollama_async_client_kwargs(host=host))

    @staticmethod
    def _is_retryable_response_error(error: ResponseError) -> bool:
        return getattr(error, "status_code", 0) in {429, 500, 502, 503, 504}

    @staticmethod
    def _extract_tool_call_fields(part: Any) -> tuple[Optional[str], Any]:
        """Best-effort extraction of tool-call name/arguments across shapes."""
        if isinstance(part, FunctionCall):
            return getattr(part, "name", None), getattr(part, "arguments", None)

        if isinstance(part, tuple) and len(part) >= 2:
            # Common fallback shape: (name, arguments)
            name = part[0] if isinstance(part[0], str) else None
            args = part[1]
            return name, args

        if isinstance(part, Mapping):
            # Support both {'name': ..., 'arguments': ...} and OpenAI-style
            # {'function': {'name': ..., 'arguments': ...}}
            if "function" in part and isinstance(part.get("function"), Mapping):
                fn = part["function"]
                return fn.get("name"), fn.get("arguments")
            return part.get("name"), part.get("arguments")

        return None, None

    @staticmethod
    def _normalize_tool_obj(tool: Any) -> Any:
        """Unwrap tuple-wrapped tools to their underlying tool/dict object."""
        if isinstance(tool, tuple):
            for item in tool:
                if hasattr(item, "schema") or isinstance(item, Mapping):
                    return item
            if tool:
                return tool[0]
        return tool

    @staticmethod
    def _response_tool_call_fields(tc: Any) -> tuple[str, str, Any]:
        """Extract (id, name, arguments) from Ollama tool-call shapes safely."""
        # Object-like tc.function.name / tc.function.arguments
        tc_function = getattr(tc, "function", None)
        if tc_function is not None:
            name = getattr(tc_function, "name", None)
            args = getattr(tc_function, "arguments", None)
            if isinstance(tc_function, Mapping):
                name = tc_function.get("name", name)
                args = tc_function.get("arguments", args)
            if name:
                call_id = getattr(tc, "id", None) or ("call_" + str(hash(name)))
                return call_id, name, args if args is not None else {}

        # Dict-like tool call
        if isinstance(tc, Mapping):
            fn = tc.get("function", {})
            if isinstance(fn, Mapping):
                name = fn.get("name")
                args = fn.get("arguments", {})
            else:
                name = tc.get("name")
                args = tc.get("arguments", {})
            if name:
                call_id = tc.get("id") or ("call_" + str(hash(name)))
                return call_id, name, args

        # Tuple-like fallback: (name, args) or (id, name, args)
        if isinstance(tc, tuple):
            if len(tc) >= 3 and isinstance(tc[1], str):
                call_id = str(tc[0]) if tc[0] else ("call_" + str(hash(tc[1])))
                return call_id, tc[1], tc[2]
            if len(tc) >= 2 and isinstance(tc[0], str):
                call_id = "call_" + str(hash(tc[0]))
                return call_id, tc[0], tc[1]

        raise ValueError(f"Unsupported tool call shape: {type(tc)}")

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Optional[Sequence[Union[Tool, Mapping[str, Any]]]] = None,
        **kwargs
    ) -> CreateResult:
        # 1. Convert AutoGen messages to Ollama format
        ollama_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AssistantMessage) :
                role = "assistant"
                # Handle tool calls in assistant messages
                tool_calls = []
                if msg.content and isinstance(msg.content, list):
                    for part in msg.content:
                        name, arguments = self._extract_tool_call_fields(part)
                        if not name:
                            continue
                        parsed_args = arguments
                        if isinstance(parsed_args, str):
                            try:
                                parsed_args = json.loads(parsed_args)
                            except json.JSONDecodeError:
                                # Keep raw string if it's not JSON.
                                pass
                        if parsed_args is None:
                            parsed_args = {}

                        if name:
                            tool_calls.append({
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": parsed_args
                                }
                            })
                
                msg_dict = {"role": role}
                if isinstance(msg.content, str):
                    msg_dict["content"] = msg.content
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                ollama_messages.append(msg_dict)
                continue
            elif isinstance(msg, UserMessage):
                role = "user"
            elif isinstance(msg, FunctionExecutionResultMessage):
                role = "tool"
                content = msg.content
                if isinstance(content, list):
                    content = " ".join([str(c) for c in content])
                ollama_messages.append({"role": role, "content": content})
                continue
            
            content = msg.content
            if isinstance(content, list):
                content = " ".join([str(c) for c in content])
            ollama_messages.append({"role": role, "content": content})

        # 2. Convert AutoGen tools to Ollama format
        ollama_tools = []
        if tools:
            for tool in tools:
                tool = self._normalize_tool_obj(tool)
                if hasattr(tool, 'schema') and callable(tool.schema):
                    s = tool.schema()
                    name = getattr(tool, "name", "unknown")
                    description = getattr(tool, "description", "")
                    parameters = s.get("parameters", {})
                elif isinstance(tool, dict):
                    name = tool.get("name", "unknown")
                    description = tool.get("description", "")
                    parameters = tool.get("parameters", {})
                else:
                    # Fallback for unexpected types
                    name = getattr(tool, 'name', 'unknown')
                    description = getattr(tool, 'description', '')
                    parameters = {}

                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters
                    }
                })

        # 3. Call Ollama SDK
        max_retries = max(0, OLLAMA_MAX_RETRIES)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await asyncio.wait_for(
                    self._client.chat(
                        model=self._model,
                        messages=ollama_messages,
                        tools=ollama_tools if ollama_tools else None,
                    ),
                    timeout=OLLAMA_REQUEST_TIMEOUT_SEC,
                )
                break
            except asyncio.TimeoutError as e:
                if attempt <= max_retries:
                    delay = OLLAMA_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Ollama chat timed out after {OLLAMA_REQUEST_TIMEOUT_SEC:.1f}s "
                    f"for model '{self._model}' after {attempt} attempts."
                ) from e
            except ResponseError as e:
                if e.status_code == 404:
                    raise ResponseError(
                        f"{e.error} — For https://ollama.com, set OLLAMA_MODEL to a name from "
                        "`GET /api/tags` with your API key. For local Ollama, use "
                        "OLLAMA_BASE_URL=http://localhost:11434 and `ollama pull <model>`.",
                        e.status_code,
                    ) from e

                if self._is_retryable_response_error(e) and attempt <= max_retries:
                    delay = OLLAMA_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                    continue

                raise RuntimeError(
                    f"Ollama request failed for model '{self._model}' "
                    f"after {attempt} attempts (status {e.status_code}): {e}"
                ) from e

        # 4. Handle tool calls in the response
        content = response.message.content or ""
        tool_calls_result = []
        if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            for tc in response.message.tool_calls:
                try:
                    call_id, call_name, call_args = self._response_tool_call_fields(tc)
                except ValueError:
                    continue

                tool_calls_result.append(FunctionCall(
                    id=call_id,
                    name=call_name,
                    arguments=json.dumps(call_args)
                ))

        # 5. Format result
        return CreateResult(
            finish_reason="stop" if not tool_calls_result else "function_calls",
            content=tool_calls_result if tool_calls_result else content,
            usage=RequestUsage(
                prompt_tokens=getattr(response, 'prompt_eval_count', 0),
                completion_tokens=getattr(response, 'eval_count', 0)
            ),
            cached=False
        )

    async def create_stream(self, messages: Sequence[LLMMessage], tools: Optional[Sequence[Tool]] = None, **kwargs):
        result = await self.create(messages, tools, **kwargs)
        yield result

    def remaining_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool] = []) -> int:
        return 32000

    def count_tokens(self, messages: Sequence[LLMMessage], tools: Sequence[Tool] = []) -> int:
        return 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            vision=False, 
            function_calling=True, 
            json_output=True
        )

    @property
    def model_info(self) -> Mapping[str, Any]:
        return {
            "model": self._model,
            "vision": False,
            "function_calling": True,
            "family": "other",
        }

    @property
    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def close(self) -> None:
        pass
