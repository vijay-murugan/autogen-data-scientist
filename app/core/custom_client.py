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

from app.core.config import ollama_async_client_kwargs


class SimpleOllamaClient(ChatCompletionClient):
    """
    A simple wrapper around the official Ollama Python SDK for AutoGen 0.4.
    Supports tool calling for code execution.
    """
    def __init__(self, model: str, host: str, **kwargs):
        self._model = model
        self._client = AsyncClient(**ollama_async_client_kwargs(host=host))

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
                        if isinstance(part, FunctionCall):
                            tool_calls.append({
                                "type": "function",
                                "function": {
                                    "name": part.name,
                                    "arguments": json.loads(part.arguments) if isinstance(part.arguments, str) else part.arguments
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
                if hasattr(tool, 'schema') and callable(tool.schema):
                    s = tool.schema()
                    name = tool.name
                    description = tool.description
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
        try:
            response = await self._client.chat(
                model=self._model,
                messages=ollama_messages,
                tools=ollama_tools if ollama_tools else None,
            )
        except ResponseError as e:
            if e.status_code == 404:
                raise ResponseError(
                    f"{e.error} — For https://ollama.com, set OLLAMA_MODEL to a name from "
                    "`GET /api/tags` with your API key. For local Ollama, use "
                    "OLLAMA_BASE_URL=http://localhost:11434 and `ollama pull <model>`.",
                    e.status_code,
                ) from e
            raise

        # 4. Handle tool calls in the response
        content = response.message.content or ""
        tool_calls_result = []
        if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            for tc in response.message.tool_calls:
                tool_calls_result.append(FunctionCall(
                    id=getattr(tc, 'id', "call_" + str(hash(tc.function.name))),
                    name=tc.function.name,
                    arguments=json.dumps(tc.function.arguments)
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
