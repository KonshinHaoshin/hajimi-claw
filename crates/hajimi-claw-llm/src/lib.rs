use async_stream::try_stream;
use hajimi_claw_types::{
    AgentEvent, AgentRequest, AgentStream, ClawError, ClawResult, ConversationMessage, LlmBackend,
    MessageRole,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleBackend {
    client: Client,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAiCompatibleBackend {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
            model,
        }
    }
}

#[async_trait::async_trait]
impl LlmBackend for OpenAiCompatibleBackend {
    async fn respond(&self, req: AgentRequest) -> ClawResult<AgentStream> {
        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
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
    use chrono::Utc;
    use futures::TryStreamExt;
    use hajimi_claw_types::{
        AgentRequest, ConversationId, ConversationMessage, LlmBackend, MessageRole,
    };

    use super::StaticBackend;

    #[tokio::test]
    async fn static_backend_streams_response() {
        let backend = StaticBackend::new("ok");
        let stream = backend
            .respond(AgentRequest {
                conversation_id: ConversationId::new(),
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
}
