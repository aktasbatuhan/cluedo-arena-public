Title: Quick Start | OpenRouter

URL Source: https://openrouter.ai/docs/quick-start

Markdown Content:
Quick Start
-----------

OpenRouter provides an OpenAI-compatible completion API to 0 models & providers that you can call directly, or using the OpenAI SDK. Additionally, some third-party SDKs are available.

In the examples below, the [OpenRouter-specific headers](https://openrouter.ai/docs/requests#request-headers) are optional. Setting them allows your app to appear on the OpenRouter leaderboards.

Using the OpenAI SDK
--------------------



import OpenAI from “openai”

const openai = new OpenAI({
baseURL: “https://openrouter.ai/api/v1”,
apiKey: “<OPENROUTER_API_KEY>”,
defaultHeaders: {
“HTTP-Referer”: “<YOUR_SITE_URL>”, // Optional. Site URL for rankings on openrouter.ai.
“X-Title”: “<YOUR_SITE_NAME>”, // Optional. Site title for rankings on openrouter.ai.
}
})

async function main() {
const completion = await openai.chat.completions.create({
model: “openai/gpt-3.5-turbo”,
messages: [
{
“role”: “user”,
“content”: “What is the meaning of life?”
}
]
})

console.log(completion.choices[0].message)
}
main()


Using the OpenRouter API directly
---------------------------------



fetch(“https://openrouter.ai/api/v1/chat/completions”, {
method: “POST”,
headers: {
“Authorization”: “Bearer <OPENROUTER_API_KEY>”,
“HTTP-Referer”: “<YOUR_SITE_URL>”, // Optional. Site URL for rankings on openrouter.ai.
“X-Title”: “<YOUR_SITE_NAME>”, // Optional. Site title for rankings on openrouter.ai.
“Content-Type”: “application/json”
},
body: JSON.stringify({
“model”: “openai/gpt-3.5-turbo”,
“messages”: [
{
“role”: “user”,
“content”: “What is the meaning of life?”
}
]
})
});
