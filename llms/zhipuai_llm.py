from __future__ import annotations

import asyncio
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)

import logging
from langchain_core.language_models.chat_models import(
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.pydantic_v1 import BaseModel, Field,root_validator

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.utils import(
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)

from packaging.version import parse
from importlib.metadata import version

def _convert_message_to_dict(message:BaseMessage)->Dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    elif role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


class ZhipuAI(BaseChatModel):
    """`ZhipuAI` Chat large language models API.

    To use, you should have the environment variable ``ZHIPUAI_API_KEY``
    set with your API key, or pass it as a named parameter to the constructor.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model_name="gpt-3.5-turbo")
    """
    client: Any = Field(default=None,exclude=True)
    
    zhipuai_api_key: Optional[str] = Field(default=None,alias="api_key")
    
    model_name: str = Field("glm-3-turbo",alias="model")
    """
    Model name to use.
    -glm-4
    -glm-4v
    -glm-3-turbo
    -cogview-3
    -chatglm-3
    -chatglm-lite
    -chatglm-pro
    -chatglm-std
    """

    temperature: float = Field(0.95)

    top_p: float = Field(0.7)

    request_id: Optional[str] = Field(None)

    do_sample: Optional[bool] = Field(True)

    streaming: bool = Field(False)

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    max_tokens: Optional[int] = None

    max_retries: int = 2

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True
    
    @root_validator()
    def validate_environment(cls,values:Dict) -> Dict:
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )
        return values
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ZhipuAI API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params
    

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) ->str:
        """Return the type of chat model."""
        return "zhipuai"
    
    @property
    def lc_secrets(self) -> Dict[str,str]:
        return{"zhipu_api_key":"ZHIPUAI_API_KEY"}
    
    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "zhipuai"]
    
    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.model_name:
            attributes["model"] = self.model_name

        if self.streaming:
            attributes["streaming"] = self.streaming

        if self.max_tokens:
            attributes["max_tokens"] = self.max_tokens

        return attributes
    
    

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the zhipuai client."""
        zhipuai_creds: Dict[str, Any] = {
            "request_id": self.request_id,
        }
        return {**self._default_params, **zhipuai_creds}
    
    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params


    def _create_chat_result(
        self, response: Union[dict, BaseModel]
    ) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)



    def completions(self, **kwargs) -> Any | None:
        print(kwargs)
        result =  self.client.chat.completions.create(**kwargs)
        print(result)
        return result

    async def async_completions(self, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        partial_func = partial(self.client.chat.completions.create, **kwargs)
        response = await loop.run_in_executor(
            None,
            partial_func,
        )
        return response
    
    async def async_completions_result(self, task_id):
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            self.client.asyncCompletions.retrieve_completion_result,
            task_id,
        )
        return response


    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        
        retry_decorator = self._create_retry_decorator(run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.completions(**kwargs)

        return _completion_with_retry(**kwargs)
    
    async def acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use tenacity to retry the async completion call."""

        retry_decorator = self._create_retry_decorator(run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            return await self.async_completions(**kwargs)

        return await _completion_with_retry(**kwargs)

    def _create_retry_decorator(
        self,
        run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
    ) -> Callable[[Any], Any]:
        import zhipuai

        errors = [
            zhipuai.ZhipuAIError,
            zhipuai.APIStatusError,
            zhipuai.APIRequestFailedError,
            zhipuai.APIReachLimitError,
            zhipuai.APIInternalError,
            zhipuai.APIServerFlowExceedError,
            zhipuai.APIResponseError,
            zhipuai.APIResponseValidationError,
            zhipuai.APITimeoutError,
        ]
        return create_base_retry_decorator(
            error_types=errors, max_retries=self.max_retries, run_manager=run_manager
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat response in chunks."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info or None
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, logprobs=logprobs)
            yield chunk
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""

        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate a chat response."""
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        response = await self.acompletion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    



    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            from zhipuai import ZhipuAI
            
            if parse(version("zhipuai")).major < 2:
                raise RuntimeError("zhipuai package version is too low"
                    "please install it via 'pip install --upgrade zhipuai'")

            self.client = ZhipuAI(
                api_key= self.zhipuai_api_key,
            )

        except ImportError:
            raise RuntimeError(
                "zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )

