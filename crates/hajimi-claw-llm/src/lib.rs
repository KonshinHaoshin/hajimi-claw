use std::sync::Arc;
use std::time::Duration;

use async_stream::try_stream;
use hajimi_claw_store::Store;
use hajimi_claw_types::{
    AgentEvent, AgentRequest, AgentStream, ClawError, ClawResult, ConversationMessage, LlmBackend,
    MessageRole, ProviderConfig, ProviderHealth, ProviderKind, ProviderRecord, ToolSpec,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleBackend {
    client: Client,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub extra_headers: Vec<(String, String)>,
}

impl OpenAiCompatibleBackend {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            client: default_http_client(),
            base_url,
            api_key,
            model,
            extra_headers: Vec::new(),
        }
    }

    pub fn from_provider(provider: &ProviderConfig) -> Self {
        Self {
            client: default_http_client(),
            base_url: provider.base_url.clone(),
            api_key: provider.api_key.clone(),
            model: provider.model.clone(),
            extra_headers: provider.extra_headers.clone(),
        }
    }
}

#[async_trait::async_trait]
impl LlmBackend for OpenAiCompatibleBackend {
    async fn respond(&self, req: AgentRequest) -> ClawResult<AgentStream> {
        let has_tools = !req.tool_specs.is_empty();
        let mut builder = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key);
        for (key, value) in &self.extra_headers {
            builder = builder.header(key, value);
        }
        let response = builder
            .json(&ChatCompletionRequest {
                model: self.model.clone(),
                messages: flatten_messages(req.system_prompt, req.messages, req.tool_history),
                tools: has_tools.then(|| map_tool_specs(&req.tool_specs)),
                tool_choice: has_tools.then_some("auto".into()),
                stream: false,
            })
            .send()
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;

        if !response.status().is_success() {
            return Err(ClawError::Backend(format!(
                "llm request failed with status {}",
                response.status()
            )));
        }

        let body: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let choice = body.choices.first().cloned().unwrap_or(Choice {
            message: ChatCompletionMessage {
                content: None,
                tool_calls: None,
            },
        });
        let text = choice.message.content.unwrap_or_default();
        let tool_calls = choice.message.tool_calls.unwrap_or_default();

        let stream = try_stream! {
            if !text.is_empty() {
                yield AgentEvent::TextDelta(text);
            }
            for tool_call in tool_calls {
                let input = serde_json::from_str(&tool_call.function.arguments)
                    .map_err(|err| ClawError::Backend(format!("decode tool arguments: {err}")))?;
                yield AgentEvent::ToolCall {
                    id: Some(tool_call.id),
                    tool: tool_call.function.name,
                    input,
                };
            }
            yield AgentEvent::Finished;
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Debug, Clone)]
pub struct StaticBackend {
    response: String,
}

impl StaticBackend {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

#[async_trait::async_trait]
impl LlmBackend for StaticBackend {
    async fn respond(&self, _req: AgentRequest) -> ClawResult<AgentStream> {
        let text = self.response.clone();
        let stream = try_stream! {
            yield AgentEvent::TextDelta(text);
            yield AgentEvent::Finished;
        };
        Ok(Box::pin(stream))
    }
}

pub struct StoreBackedBackend {
    store: Arc<Store>,
    fallback: Option<Arc<dyn LlmBackend>>,
}

impl StoreBackedBackend {
    pub fn new(store: Arc<Store>, fallback: Option<Arc<dyn LlmBackend>>) -> Self {
        Self { store, fallback }
    }
}

#[async_trait::async_trait]
impl LlmBackend for StoreBackedBackend {
    async fn respond(&self, req: AgentRequest) -> ClawResult<AgentStream> {
        let provider = resolve_provider(&self.store, req.provider_id.as_deref())
            .map_err(|err| ClawError::Backend(err.to_string()))?;

        match provider {
            Some(record) if record.config.enabled => match record.config.kind {
                ProviderKind::OpenAiCompatible | ProviderKind::CustomChatCompletions => {
                    respond_with_fallback_models(&record.config, req).await
                }
            },
            Some(_) => Err(ClawError::Backend("provider is disabled".into())),
            None => match &self.fallback {
                Some(fallback) => fallback.respond(req).await,
                None => Err(ClawError::NotFound(
                    "no configured provider and no fallback backend".into(),
                )),
            },
        }
    }
}

async fn respond_with_fallback_models(
    provider: &ProviderConfig,
    req: AgentRequest,
) -> ClawResult<AgentStream> {
    let mut models = vec![provider.model.clone()];
    for model in &provider.fallback_models {
        if !model.trim().is_empty() && !models.iter().any(|item| item == model) {
            models.push(model.clone());
        }
    }

    let mut last_error = None;
    for model in models {
        let mut attempt = provider.clone();
        attempt.model = model.clone();
        match OpenAiCompatibleBackend::from_provider(&attempt)
            .respond(req.clone())
            .await
        {
            Ok(stream) => return Ok(stream),
            Err(err) => {
                last_error = Some((model, err));
            }
        }
    }

    match last_error {
        Some((model, err)) => Err(ClawError::Backend(format!(
            "provider failed after trying primary/fallback models; last model `{model}`: {err}"
        ))),
        None => Err(ClawError::Backend(
            "provider has no primary or fallback models configured".into(),
        )),
    }
}

pub async fn test_provider(
    client: &Client,
    provider: &ProviderConfig,
) -> ClawResult<ProviderHealth> {
    let suggested_models = match list_models(client, provider).await {
        Ok(models) => models,
        Err(ClawError::Backend(message)) => {
            return Ok(ProviderHealth {
                ok: false,
                message,
                suggested_models: vec![provider.model.clone()],
            });
        }
        Err(err) => return Err(err),
    };
    let model_ok = suggested_models.iter().any(|item| item == &provider.model);

    Ok(ProviderHealth {
        ok: true,
        message: if model_ok {
            format!("connected; model `{}` is available", provider.model)
        } else if suggested_models.is_empty() {
            format!(
                "connected; model list unavailable, keeping `{}`",
                provider.model
            )
        } else {
            format!(
                "connected; configured model `{}` not listed, examples: {}",
                provider.model,
                suggested_models.join(", ")
            )
        },
        suggested_models,
    })
}

pub async fn list_models(client: &Client, provider: &ProviderConfig) -> ClawResult<Vec<String>> {
    let models_url = format!("{}/models", provider.base_url.trim_end_matches('/'));
    let mut builder = client.get(models_url).bearer_auth(&provider.api_key);
    for (key, value) in &provider.extra_headers {
        builder = builder.header(key, value);
    }
    let response = builder
        .send()
        .await
        .map_err(|err| ClawError::Backend(err.to_string()))?;

    if !response.status().is_success() {
        return Err(ClawError::Backend(format!(
            "provider returned {}",
            response.status()
        )));
    }

    let payload: ModelsResponse = response
        .json()
        .await
        .map_err(|err| ClawError::Backend(err.to_string()))?;
    Ok(payload
        .data
        .into_iter()
        .map(|item| item.id)
        .take(8)
        .collect::<Vec<_>>())
}

fn resolve_provider(
    store: &Store,
    provider_id: Option<&str>,
) -> anyhow::Result<Option<ProviderRecord>> {
    if let Some(provider_id) = provider_id {
        return store.get_provider(provider_id);
    }
    store
        .get_default_provider()
        .and_then(|record| match record {
            Some(record) => Ok(Some(record)),
            None => store.get_first_provider(),
        })
}

fn default_http_client() -> Client {
    Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .build()
        .expect("build reqwest client")
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiToolSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Serialize, Clone)]
struct OpenAiToolSpec {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunctionSpec,
}

#[derive(Debug, Serialize, Clone)]
struct OpenAiFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize, Clone)]
struct Choice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize, Clone)]
struct ChatCompletionMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type", default)]
    tool_type: String,
    function: OpenAiFunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelItem>,
}

#[derive(Debug, Deserialize)]
struct ModelItem {
    id: String,
}

fn flatten_messages(
    system_prompt: String,
    messages: Vec<ConversationMessage>,
    tool_history: Vec<hajimi_claw_types::ToolExchange>,
) -> Vec<ChatMessage> {
    let mut flattened = vec![ChatMessage {
        role: "system".into(),
        content: Some(system_prompt),
        tool_call_id: None,
        tool_calls: None,
    }];
    flattened.extend(messages.into_iter().map(|message| {
        ChatMessage {
            role: match message.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
            }
            .into(),
            content: Some(message.content),
            tool_call_id: None,
            tool_calls: None,
        }
    }));
    for exchange in tool_history {
        flattened.push(ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_call_id: None,
            tool_calls: Some(vec![OpenAiToolCall {
                id: exchange
                    .call
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("call_{}", exchange.call.name)),
                tool_type: "function".into(),
                function: OpenAiFunctionCall {
                    name: exchange.call.name,
                    arguments: serde_json::to_string(&exchange.call.arguments)
                        .unwrap_or_else(|_| "{}".into()),
                },
            }]),
        });
        flattened.push(ChatMessage {
            role: "tool".into(),
            content: Some(exchange.result.content),
            tool_call_id: exchange.result.call_id,
            tool_calls: None,
        });
    }
    flattened
}

fn map_tool_specs(specs: &[ToolSpec]) -> Vec<OpenAiToolSpec> {
    specs
        .iter()
        .map(|spec| OpenAiToolSpec {
            tool_type: "function".into(),
            function: OpenAiFunctionSpec {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: spec.input_schema.clone(),
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use futures::TryStreamExt;
    use hajimi_claw_store::Store;
    use hajimi_claw_types::{
        AgentRequest, ConversationId, ConversationMessage, LlmBackend, MessageRole, ProviderConfig,
        ProviderKind, ProviderRecord,
    };

    use super::{StaticBackend, StoreBackedBackend};

    #[tokio::test]
    async fn static_backend_streams_response() {
        let backend = StaticBackend::new("ok");
        let stream = backend
            .respond(AgentRequest {
                conversation_id: ConversationId::new(),
                provider_id: None,
                system_prompt: "system".into(),
                messages: vec![ConversationMessage {
                    role: MessageRole::User,
                    content: "hello".into(),
                    created_at: Utc::now(),
                }],
                tool_specs: vec![],
                tool_history: vec![],
            })
            .await
            .unwrap();
        let events = stream.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(events.len(), 2);
    }

    #[tokio::test]
    async fn store_backed_backend_uses_fallback_without_provider() {
        let store = Arc::new(Store::open_in_memory().unwrap());
        let backend =
            StoreBackedBackend::new(store, Some(Arc::new(StaticBackend::new("fallback"))));
        let stream = backend
            .respond(AgentRequest {
                conversation_id: ConversationId::new(),
                provider_id: None,
                system_prompt: "system".into(),
                messages: vec![ConversationMessage {
                    role: MessageRole::User,
                    content: "hello".into(),
                    created_at: Utc::now(),
                }],
                tool_specs: vec![],
                tool_history: vec![],
            })
            .await
            .unwrap();
        let events = stream.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn provider_backend_can_be_stored() {
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_provider(&ProviderRecord {
                config: ProviderConfig {
                    id: "demo".into(),
                    label: "Demo".into(),
                    kind: ProviderKind::OpenAiCompatible,
                    base_url: "https://example.com/v1".into(),
                    api_key: "secret".into(),
                    model: "gpt-demo".into(),
                    fallback_models: vec![],
                    enabled: true,
                    extra_headers: vec![],
                    created_at: Utc::now(),
                },
                is_default: true,
            })
            .unwrap();
        assert!(store.get_default_provider().unwrap().is_some());
    }

    #[test]
    fn maps_tool_specs_to_openai_functions() {
        let specs = vec![hajimi_claw_types::ToolSpec {
            name: "exec_once".into(),
            description: "Run one command.".into(),
            requires_approval: true,
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        }];

        let mapped = super::map_tool_specs(&specs);
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].tool_type, "function");
        assert_eq!(mapped[0].function.name, "exec_once");
    }
}
