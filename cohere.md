### Request
POST https://api.cohere.com/v2/chat

import cohere

co = cohere.ClientV2()

response = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "hello world!"}],
)

print(response)

### Response
{
  "id": "c14c80c3-18eb-4519-9460-6c92edd8cfb4",
  "finish_reason": "COMPLETE",
  "message": {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "LLMs stand for Large Language Models, which are a type of neural network model specialized in processing and generating human language. They are designed to understand and respond to natural language input and have become increasingly popular and valuable in recent years.\n\nLLMs are trained on vast amounts of text data, enabling them to learn patterns, grammar, and semantic meanings present in the language. These models can then be used for various natural language processing tasks, such as text generation, summarization, question answering, machine translation, sentiment analysis, and even some aspects of natural language understanding.\n\nSome well-known examples of LLMs include:\n\n1. GPT-3 (Generative Pre-trained Transformer 3) — An open-source LLM developed by OpenAI, capable of generating human-like text and performing various language tasks.\n\n2. BERT (Bidirectional Encoder Representations from Transformers) — A Google-developed LLM that is particularly good at understanding contextual relationships in text, and is widely used for natural language understanding tasks like sentiment analysis and named entity recognition.\n\n3. T5 (Text-to-Text Transfer Transformer) — Also from Google, T5 is a flexible LLM that frames all language tasks as text-to-text problems, where the model learns to generate output text based on input text prompts.\n\n4. RoBERTa (Robustly Optimized BERT Approach) — A variant of BERT that uses additional training techniques to improve performance.\n\n5. DeBERTa (Decoding-enhanced BERT with disentangled attention) — Another variant of BERT that introduces a new attention mechanism.\n\nLLMs have become increasingly powerful and larger in scale, improving the accuracy and sophistication of language tasks. They are also being used as a foundation for developing various applications, including chatbots, content recommendation systems, language translation services, and more. \n\nThe future of LLMs holds the potential for even more sophisticated language technologies, with ongoing research and development focused on enhancing their capabilities, improving efficiency, and exploring their applications in various domains."
      }
    ]
  },
  "usage": {
    "billed_units": {
      "input_tokens": 5,
      "output_tokens": 418
    },
    "tokens": {
      "input_tokens": 71,
      "output_tokens": 418
    }
  }
}



### Headers

X-Client-NamestringOptional

The name of the project that is making the request.

### Request

This endpoint expects an object.

streamfalseRequired

Defaults to `false`.

When `true`, the response will be a SSE stream of events. The final event will contain the complete response, and will have an `event_type` of `"stream-end"`.

Streaming is beneficial for user interfaces that render the contents of the response piece by piece, as it gets generated.

modelstringRequired

The name of a compatible [Cohere model](https://docs.cohere.com/v2/docs/models) or the ID of a [fine-tuned](https://docs.cohere.com/v2/docs/chat-fine-tuning) model.

messageslist of objectsRequired

A list of chat messages in chronological order, representing a conversation between the user and the model.

Messages can be from `User`, `Assistant`, `Tool` and `System` roles. Learn more about messages and roles in [the Chat API guide](https://docs.cohere.com/v2/docs/chat-api).

Show 4 variants

toolslist of objectsOptional

A list of available tools (functions) that the model may suggest invoking before producing a text response.

When `tools` is passed (without `tool_results`), the `text` content in the response will be empty and the `tool_calls` field in the response will be populated with a list of tool calls that need to be made. If no calls need to be made, the `tool_calls` array will be empty.

Show 2 properties

documentslist of strings or objectsOptional

A list of relevant documents that the model can cite to generate a more accurate reply. Each document is either a string or document object with content and metadata.

Show 2 variants

citation\_optionsobjectOptional

Options for controlling citation generation.

Show property

response\_formatobjectOptional

Configuration for forcing the model output to adhere to the specified format. Supported on [Command R](https://docs.cohere.com/v2/docs/command-r), [Command R+](https://docs.cohere.com/v2/docs/command-r-plus) and newer models.

The model can be forced into outputting JSON objects by setting `{ "type": "json_object" }`.

A [JSON Schema](https://json-schema.org/) can optionally be provided, to ensure a specific structure.

**Note**: When using `{ "type": "json_object" }` your `message` should always explicitly instruct the model to generate a JSON (eg: _“Generate a JSON …”_) . Otherwise the model may end up getting stuck generating an infinite stream of characters and eventually run out of context length.

**Note**: When `json_schema` is not specified, the generated object can have up to 5 layers of nesting.

**Limitation**: The parameter is not supported when used in combinations with the `documents` or `tools` parameters.

Show 2 variants

safety\_modeenumOptional

Allowed values: CONTEXTUALSTRICTOFF

Used to select the [safety instruction](https://docs.cohere.com/v2/docs/safety-modes) inserted into the prompt. Defaults to `CONTEXTUAL`.
When `OFF` is specified, the safety instruction will be omitted.

Safety modes are not yet configurable in combination with `tools`, `tool_results` and `documents` parameters.

**Note**: This parameter is only compatible newer Cohere models, starting with [Command R 08-2024](https://docs.cohere.com/docs/command-r#august-2024-release) and [Command R+ 08-2024](https://docs.cohere.com/docs/command-r-plus#august-2024-release).

**Note**: `command-r7b-12-2024` and newer models only support `"CONTEXTUAL"` and `"STRICT"` modes.

max\_tokensintegerOptional

The maximum number of tokens the model will generate as part of the response.

**Note**: Setting a low value may result in incomplete generations.

stop\_sequenceslist of stringsOptional

A list of up to 5 strings that the model will use to stop generation. If the model generates a string that matches any of the strings in the list, it will stop generating tokens and return the generated text up to that point not including the stop sequence.

temperaturedoubleOptional

Defaults to `0.3`.

A non-negative float that tunes the degree of randomness in generation. Lower temperatures mean less random generations, and higher temperatures mean more random generations.

Randomness can be further maximized by increasing the value of the `p` parameter.

seedintegerOptional

If specified, the backend will make a best effort to sample tokens
deterministically, such that repeated requests with the same
seed and parameters should return the same result. However,
determinism cannot be totally guaranteed.

frequency\_penaltydoubleOptional

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.
Used to reduce repetitiveness of generated tokens. The higher the value, the stronger a penalty is applied to previously present tokens, proportional to how many times they have already appeared in the prompt or prior generation.

presence\_penaltydoubleOptional

Defaults to `0.0`, min value of `0.0`, max value of `1.0`.
Used to reduce repetitiveness of generated tokens. Similar to `frequency_penalty`, except that this penalty is applied equally to all tokens that have already appeared, regardless of their exact frequencies.

kdoubleOptional

Ensures that only the top `k` most likely tokens are considered for generation at each step. When `k` is set to `0`, k-sampling is disabled.
Defaults to `0`, min value of `0`, max value of `500`.

pdoubleOptional

Ensures that only the most likely tokens, with total probability mass of `p`, are considered for generation at each step. If both `k` and `p` are enabled, `p` acts after `k`.
Defaults to `0.75`. min value of `0.01`, max value of `0.99`.

logprobsbooleanOptional

Defaults to `false`. When set to `true`, the log probabilities of the generated tokens will be included in the response.

tool\_choiceenumOptional

Allowed values: REQUIREDNONE

Used to control whether or not the model will be forced to use a tool when answering. When `REQUIRED` is specified, the model will be forced to use at least one of the user-defined tools, and the `tools` parameter must be passed in the request.
When `NONE` is specified, the model will be forced **not** to use one of the specified tools, and give a direct response.
If tool\_choice isn’t specified, then the model is free to choose whether to use the specified tools or not.

**Note**: This parameter is only compatible with models [Command-r7b](https://docs.cohere.com/v2/docs/command-r7b) and newer.

**Note**: The same functionality can be achieved in `/v1/chat` using the `force_single_step` parameter. If `force_single_step=true`, this is equivalent to specifying `REQUIRED`. While if `force_single_step=true` and `tool_results` are passed, this is equivalent to specifying `NONE`.

strict\_toolsbooleanOptionalBeta

When set to `true`, tool calls in the Assistant message will be forced to follow the tool definition strictly. Learn more in the [Structured Outputs (Tools) guide](https://docs.cohere.com/docs/structured-outputs-json#structured-outputs-tools).

**Note**: The first few requests with a new set of tools will take longer to process.

### Response

idstring

Unique identifier for the generated reply. Useful for submitting feedback.

finish\_reasonenum

Allowed values: COMPLETESTOP\_SEQUENCEMAX\_TOKENSTOOL\_CALLERROR

The reason a chat request has finished.

- **complete**: The model finished sending a complete message.
- **max\_tokens**: The number of generated tokens exceeded the model’s context length or the value specified via the `max_tokens` parameter.
- **stop\_sequence**: One of the provided `stop_sequence` entries was reached in the model’s generation.
- **tool\_call**: The model generated a Tool Call and is expecting a Tool Message in return
- **error**: The generation failed due to an internal error

messageobject

A message from the assistant role can contain text and tool call information.

Show 5 properties

usageobjectOptional

Show 2 properties

logprobslist of objectsOptional

Show 3 properties

### Errors

400

V2chat Request Bad Request Error

401

V2chat Request Unauthorized Error

403

V2chat Request Forbidden Error

404

V2chat Request Not Found Error

422

V2chat Request Unprocessable Entity Error

429

V2chat Request Too Many Requests Error

498

V2chat Request Invalid Token Error

499

V2chat Request Client Closed Request Error

500

V2chat Request Internal Server Error

501

V2chat Request Not Implemented Error

503

V2chat Request Service Unavailable Error

504

V2chat Request Gateway Timeout Error

[Chat with Streaming\\
\\
Up Next](https://docs.cohere.com/reference/chat-stream)

[Built with](https://buildwithfern.com/?utm_campaign=buildWith&utm_medium=docs&utm_source=docs.cohere.com)