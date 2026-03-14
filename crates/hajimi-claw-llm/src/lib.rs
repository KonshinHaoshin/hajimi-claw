use std::sync::Arc;

use async_stream::try_stream;
use hajimi_claw_store::Store;
use hajimi_claw_types::{
    AgentEvent, AgentRequest, AgentStream, ClawError, ClawResult, ConversationMessage, LlmBackend,
    MessageRole, ProviderConfig, ProviderHealth, ProviderKind, ProviderRecord,
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
            client: Client::new(),
            base_url,
            api_key,
            model,
            extra_headers: Vec::new(),
        }
    }

    pub fn from_provider(provider: &ProviderConfig) -> Self {
        Self {
            client: Client::new(),
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
                messages: flatten_messages(req.system_prompt, req.messages),
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
        let text = body
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();

        let stream = try_stream! {
            yield AgentEvent::TextDelta(text);
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
                    OpenAiCompatibleBackend::from_provider(&record.config)
                        .respond(req)
                        .await
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
    store.get_default_provider()
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelItem>,
}

#[derive(Debug, Deserialize)]
struct ModelItem {
    id: String,
}

fn flatten_messages(system_prompt: String, messages: Vec<ConversationMessage>) -> Vec<ChatMessage> {
    let mut flattened = vec![ChatMessage {
        role: "system".into(),
        content: system_prompt,
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
            content: message.content,
        }
    }));
    flattened
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
                    enabled: true,
                    extra_headers: vec![],
                    created_at: Utc::now(),
                },
                is_default: true,
            })
            .unwrap();
        assert!(store.get_default_provider().unwrap().is_some());
    }
}
