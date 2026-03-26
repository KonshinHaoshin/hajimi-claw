use std::collections::BTreeSet;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use hajimi_claw_types::{
    CapabilityId, CapabilityKind, ClawError, ClawResult, McpServerConfig, McpServerStatus, Tool,
    ToolContext, ToolOutput, ToolSpec,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const DEFAULT_STARTUP_TIMEOUT_SECS: u64 = 10;
const DEFAULT_TOOL_CALL_TIMEOUT_SECS: u64 = 120;
const STDERR_TAIL_BYTES: usize = 4096;

pub struct McpBootstrapResult {
    pub tools: Vec<Arc<dyn Tool>>,
    pub statuses: Vec<McpServerStatus>,
}

pub async fn bootstrap_mcp_servers(configs: &[McpServerConfig]) -> McpBootstrapResult {
    bootstrap_mcp_servers_with_connector(configs, &StdioMcpConnector).await
}

async fn bootstrap_mcp_servers_with_connector<C>(
    configs: &[McpServerConfig],
    connector: &C,
) -> McpBootstrapResult
where
    C: McpConnector,
{
    let mut tools = Vec::new();
    let mut statuses = Vec::new();
    let mut seen_local_names = BTreeSet::new();

    for config in configs {
        if !config.enabled {
            statuses.push(McpServerStatus {
                name: config.name.clone(),
                connected: false,
                tool_count: 0,
                message: "disabled in config".into(),
            });
            continue;
        }

        match connector.connect(config).await {
            Ok(server) => {
                let (registered, duplicates) = register_server_tools(
                    config,
                    server.client,
                    server.tools,
                    &mut seen_local_names,
                );
                let tool_count = registered.len();
                tools.extend(registered);
                statuses.push(McpServerStatus {
                    name: config.name.clone(),
                    connected: true,
                    tool_count,
                    message: if duplicates == 0 {
                        "connected".into()
                    } else {
                        format!("connected; skipped {duplicates} duplicate tool(s)")
                    },
                });
            }
            Err(err) => statuses.push(McpServerStatus {
                name: config.name.clone(),
                connected: false,
                tool_count: 0,
                message: err.to_string(),
            }),
        }
    }

    McpBootstrapResult { tools, statuses }
}

fn register_server_tools(
    config: &McpServerConfig,
    client: Arc<dyn McpInvoker>,
    mut remote_tools: Vec<McpRemoteTool>,
    seen_local_names: &mut BTreeSet<String>,
) -> (Vec<Arc<dyn Tool>>, usize) {
    remote_tools.sort_by(|left, right| left.name.cmp(&right.name));

    let mut registered = Vec::new();
    let mut duplicates = 0;
    for remote_tool in remote_tools {
        let adapter = Arc::new(McpToolAdapter::new(
            config.name.clone(),
            remote_tool,
            config.requires_approval,
            client.clone(),
        )) as Arc<dyn Tool>;
        let local_name = adapter.spec().name.clone();
        if seen_local_names.insert(local_name) {
            registered.push(adapter);
        } else {
            duplicates += 1;
        }
    }
    (registered, duplicates)
}

#[derive(Debug, Clone, Deserialize)]
struct McpRemoteTool {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(rename = "inputSchema", default = "default_object_schema")]
    input_schema: Value,
}

fn default_object_schema() -> Value {
    json!({
        "type": "object",
    })
}

#[derive(Debug, Deserialize)]
struct McpToolsListResult {
    #[serde(default)]
    tools: Vec<McpRemoteTool>,
    #[serde(rename = "nextCursor")]
    next_cursor: Option<String>,
}

#[derive(Debug, Deserialize)]
struct McpToolCallResult {
    #[serde(default)]
    content: Vec<Value>,
    #[serde(rename = "structuredContent")]
    structured_content: Option<Value>,
}

#[async_trait]
trait McpInvoker: Send + Sync {
    async fn call_tool(&self, tool_name: &str, input: Value) -> ClawResult<ToolOutput>;
}

struct McpConnectedServer {
    client: Arc<dyn McpInvoker>,
    tools: Vec<McpRemoteTool>,
}

#[async_trait]
trait McpConnector: Send + Sync {
    async fn connect(&self, config: &McpServerConfig) -> ClawResult<McpConnectedServer>;
}

struct StdioMcpConnector;

#[async_trait]
impl McpConnector for StdioMcpConnector {
    async fn connect(&self, config: &McpServerConfig) -> ClawResult<McpConnectedServer> {
        let (client, tools) = StdioMcpClient::spawn(config.clone()).await?;
        Ok(McpConnectedServer {
            client: Arc::new(client),
            tools,
        })
    }
}

struct McpToolAdapter {
    spec: ToolSpec,
    capability_id: CapabilityId,
    remote_name: String,
    client: Arc<dyn McpInvoker>,
}

impl McpToolAdapter {
    fn new(
        server_name: String,
        remote_tool: McpRemoteTool,
        requires_approval: bool,
        client: Arc<dyn McpInvoker>,
    ) -> Self {
        let description = if remote_tool.description.trim().is_empty() {
            format!("MCP tool `{}` from `{server_name}`", remote_tool.name)
        } else {
            remote_tool.description.clone()
        };
        let local_name = format!("mcp.{}.{}", server_name, remote_tool.name);
        Self {
            spec: ToolSpec {
                name: local_name.clone(),
                description,
                requires_approval,
                input_schema: remote_tool.input_schema,
            },
            capability_id: CapabilityId {
                kind: CapabilityKind::McpTool,
                name: local_name,
                source: Some(server_name),
            },
            remote_name: remote_tool.name,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn capability_id(&self) -> CapabilityId {
        self.capability_id.clone()
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        self.client.call_tool(&self.remote_name, input).await
    }
}

struct StdioClientState {
    _child: Child,
    transport: JsonRpcTransport<ChildStdout, ChildStdin>,
    broken: Option<String>,
}

struct StdioClientInner {
    server_name: String,
    state: Mutex<StdioClientState>,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
    tool_call_timeout: Duration,
}

#[derive(Clone)]
struct StdioMcpClient {
    inner: Arc<StdioClientInner>,
}

impl StdioMcpClient {
    async fn spawn(config: McpServerConfig) -> ClawResult<(Self, Vec<McpRemoteTool>)> {
        let mut command = Command::new(&config.command);
        command.args(&config.args);
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        command.env_clear();
        for key in &config.env_allowlist {
            if let Ok(value) = std::env::var(key) {
                command.env(key, value);
            }
        }

        let mut child = command.spawn().map_err(|err| {
            ClawError::Backend(format!("mcp server `{}` spawn failed: {err}", config.name))
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            ClawError::Backend(format!("mcp server `{}` missing stdin pipe", config.name))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ClawError::Backend(format!("mcp server `{}` missing stdout pipe", config.name))
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            ClawError::Backend(format!("mcp server `{}` missing stderr pipe", config.name))
        })?;
        let stderr_tail = spawn_stderr_collector(stderr);

        let client = Self {
            inner: Arc::new(StdioClientInner {
                server_name: config.name.clone(),
                state: Mutex::new(StdioClientState {
                    _child: child,
                    transport: JsonRpcTransport::new(stdout, stdin),
                    broken: None,
                }),
                stderr_tail,
                tool_call_timeout: Duration::from_secs(DEFAULT_TOOL_CALL_TIMEOUT_SECS),
            }),
        };

        let startup_timeout = Duration::from_secs(
            config
                .startup_timeout_secs
                .unwrap_or(DEFAULT_STARTUP_TIMEOUT_SECS),
        );
        client.initialize(startup_timeout).await?;
        let tools = client.list_tools(startup_timeout).await?;
        Ok((client, tools))
    }

    async fn initialize(&self, timeout: Duration) -> ClawResult<()> {
        let params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "hajimi-claw",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });
        let _ = self.request("initialize", Some(params), timeout).await?;
        self.notify("notifications/initialized", Some(json!({})))
            .await?;
        Ok(())
    }

    async fn list_tools(&self, timeout: Duration) -> ClawResult<Vec<McpRemoteTool>> {
        let mut cursor = None;
        let mut tools = Vec::new();
        loop {
            let params = match cursor.take() {
                Some(cursor) => Some(json!({ "cursor": cursor })),
                None => Some(json!({})),
            };
            let result = self.request("tools/list", params, timeout).await?;
            let page: McpToolsListResult = serde_json::from_value(result).map_err(|err| {
                ClawError::Backend(format!(
                    "mcp server `{}` returned invalid tools/list payload: {err}",
                    self.inner.server_name
                ))
            })?;
            tools.extend(page.tools);
            if let Some(next_cursor) = page.next_cursor.filter(|value| !value.is_empty()) {
                cursor = Some(next_cursor);
            } else {
                break;
            }
        }
        Ok(tools)
    }

    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> ClawResult<Value> {
        let mut state = self.inner.state.lock().await;
        if let Some(reason) = &state.broken {
            return Err(ClawError::Backend(reason.clone()));
        }

        match tokio::time::timeout(timeout, state.transport.request(method, params)).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(err)) => {
                let message = self.decorate_error(err.to_string()).await;
                state.broken = Some(message.clone());
                Err(ClawError::Backend(message))
            }
            Err(_) => {
                let message = self
                    .decorate_error(format!(
                        "mcp server `{}` timed out waiting for `{method}` response",
                        self.inner.server_name
                    ))
                    .await;
                state.broken = Some(message.clone());
                Err(ClawError::Backend(message))
            }
        }
    }

    async fn notify(&self, method: &str, params: Option<Value>) -> ClawResult<()> {
        let mut state = self.inner.state.lock().await;
        if let Some(reason) = &state.broken {
            return Err(ClawError::Backend(reason.clone()));
        }
        match state.transport.notify(method, params).await {
            Ok(()) => Ok(()),
            Err(err) => {
                let message = self.decorate_error(err.to_string()).await;
                state.broken = Some(message.clone());
                Err(ClawError::Backend(message))
            }
        }
    }

    async fn decorate_error(&self, detail: String) -> String {
        let stderr = self.inner.stderr_tail.lock().await.clone();
        if stderr.is_empty() {
            return detail;
        }
        let stderr = String::from_utf8_lossy(&stderr);
        let stderr = stderr.trim();
        if stderr.is_empty() {
            detail
        } else {
            format!("{detail}; stderr: {stderr}")
        }
    }
}

#[async_trait]
impl McpInvoker for StdioMcpClient {
    async fn call_tool(&self, tool_name: &str, input: Value) -> ClawResult<ToolOutput> {
        let result = self
            .request(
                "tools/call",
                Some(json!({
                    "name": tool_name,
                    "arguments": input,
                })),
                self.inner.tool_call_timeout,
            )
            .await?;
        let output: McpToolCallResult = serde_json::from_value(result).map_err(|err| {
            ClawError::Backend(format!(
                "mcp server `{}` returned invalid tools/call payload: {err}",
                self.inner.server_name
            ))
        })?;
        Ok(ToolOutput {
            content: flatten_content_blocks(&output.content),
            structured: output.structured_content,
        })
    }
}

fn flatten_content_blocks(blocks: &[Value]) -> String {
    blocks
        .iter()
        .filter_map(|block| {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                if text.trim().is_empty() {
                    None
                } else {
                    Some(text.to_string())
                }
            } else if block.is_null() {
                None
            } else {
                Some(
                    serde_json::to_string(block)
                        .unwrap_or_else(|_| String::from("{\"type\":\"unsupported\"}")),
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

struct JsonRpcTransport<R, W> {
    reader: BufReader<R>,
    writer: W,
    next_id: u64,
}

impl<R, W> JsonRpcTransport<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    fn new(reader: R, writer: W) -> Self {
        Self {
            reader: BufReader::new(reader),
            writer,
            next_id: 1,
        }
    }

    async fn request(&mut self, method: &str, params: Option<Value>) -> ClawResult<Value> {
        let request_id = self.next_id;
        self.next_id += 1;

        let mut payload = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        });
        if let Some(params) = params {
            payload["params"] = params;
        }
        write_message(&mut self.writer, &payload).await?;

        loop {
            let message = read_message(&mut self.reader).await?;
            if message.get("id") != Some(&json!(request_id)) {
                continue;
            }
            if let Some(error) = message.get("error") {
                return Err(ClawError::Backend(format_json_rpc_error(method, error)));
            }
            return message.get("result").cloned().ok_or_else(|| {
                ClawError::Backend(format!("mcp response for `{method}` missing result field"))
            });
        }
    }

    async fn notify(&mut self, method: &str, params: Option<Value>) -> ClawResult<()> {
        let mut payload = json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(params) = params {
            payload["params"] = params;
        }
        write_message(&mut self.writer, &payload).await
    }
}

async fn write_message<W>(writer: &mut W, payload: &Value) -> ClawResult<()>
where
    W: AsyncWrite + Unpin,
{
    let body = serde_json::to_vec(payload)
        .map_err(|err| ClawError::Backend(format!("serialize mcp payload: {err}")))?;
    writer
        .write_all(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes())
        .await
        .map_err(|err| ClawError::Backend(format!("write mcp headers: {err}")))?;
    writer
        .write_all(&body)
        .await
        .map_err(|err| ClawError::Backend(format!("write mcp body: {err}")))?;
    writer
        .flush()
        .await
        .map_err(|err| ClawError::Backend(format!("flush mcp stream: {err}")))?;
    Ok(())
}

async fn read_message<R>(reader: &mut R) -> ClawResult<Value>
where
    R: AsyncBufRead + Unpin,
{
    let mut content_length = None;
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .await
            .map_err(|err| ClawError::Backend(format!("read mcp header: {err}")))?;
        if read == 0 {
            return Err(ClawError::Backend("mcp transport closed".into()));
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| ClawError::Backend(format!("invalid mcp header line: {line}")))?;
        if name.eq_ignore_ascii_case("content-length") {
            content_length = Some(value.trim().parse::<usize>().map_err(|err| {
                ClawError::Backend(format!(
                    "invalid mcp content length `{}`: {err}",
                    value.trim()
                ))
            })?);
        }
    }

    let content_length = content_length
        .ok_or_else(|| ClawError::Backend("mcp message missing Content-Length header".into()))?;
    let mut body = vec![0; content_length];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|err| ClawError::Backend(format!("read mcp body: {err}")))?;
    serde_json::from_slice(&body)
        .map_err(|err| ClawError::Backend(format!("parse mcp json body: {err}")))
}

fn format_json_rpc_error(method: &str, error: &Value) -> String {
    let code = error
        .get("code")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown error");
    if let Some(data) = error.get("data") {
        format!(
            "mcp request `{method}` failed ({code}): {message}; data={}",
            serde_json::to_string(data).unwrap_or_else(|_| String::from("null"))
        )
    } else {
        format!("mcp request `{method}` failed ({code}): {message}")
    }
}

fn spawn_stderr_collector<R>(reader: R) -> Arc<Mutex<Vec<u8>>>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let tail = Arc::new(Mutex::new(Vec::new()));
    let target = tail.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(reader);
        let mut buffer = [0_u8; 1024];
        let mut collected = Vec::new();
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) => break,
                Ok(read) => {
                    collected.extend_from_slice(&buffer[..read]);
                    if collected.len() > STDERR_TAIL_BYTES {
                        let keep_from = collected.len() - STDERR_TAIL_BYTES;
                        collected.drain(..keep_from);
                    }
                }
                Err(_) => break,
            }
        }
        *target.lock().await = collected;
    });
    tail
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::{
        JsonRpcTransport, McpConnectedServer, McpRemoteTool, bootstrap_mcp_servers_with_connector,
        flatten_content_blocks, read_message, register_server_tools, write_message,
    };
    use super::{McpConnector, McpInvoker, McpToolAdapter, McpToolsListResult};
    use async_trait::async_trait;
    use hajimi_claw_types::{
        CapabilityKind, ClawError, ClawResult, McpServerConfig, Tool, ToolContext, ToolOutput,
    };
    use serde_json::{Value, json};
    use tokio::io::duplex;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn initialize_handshake_requires_initialized_notification() {
        let (client_stream, server_stream) = duplex(16 * 1024);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let (server_read, server_write) = tokio::io::split(server_stream);

        let server = tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(server_read);
            let mut writer = server_write;

            let initialize = read_message(&mut reader).await.expect("read initialize");
            assert_eq!(initialize["method"], "initialize");
            assert_eq!(initialize["params"]["protocolVersion"], "2024-11-05");
            write_message(
                &mut writer,
                &json!({
                    "jsonrpc": "2.0",
                    "id": initialize["id"].clone(),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": { "name": "demo", "version": "1.0.0" }
                    }
                }),
            )
            .await
            .expect("write initialize response");

            let initialized = read_message(&mut reader).await.expect("read initialized");
            assert_eq!(initialized["method"], "notifications/initialized");
            assert!(initialized.get("id").is_none());
        });

        let mut transport = JsonRpcTransport::new(client_read, client_write);
        let _ = transport
            .request(
                "initialize",
                Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "test", "version": "1.0.0" }
                })),
            )
            .await
            .expect("initialize succeeds");
        transport
            .notify("notifications/initialized", Some(json!({})))
            .await
            .expect("initialized notification succeeds");

        server.await.expect("server task");
    }

    #[tokio::test]
    async fn paginated_tools_list_collects_all_tools() {
        let (client_stream, server_stream) = duplex(16 * 1024);
        let (client_read, client_write) = tokio::io::split(client_stream);
        let (server_read, server_write) = tokio::io::split(server_stream);

        let server = tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(server_read);
            let mut writer = server_write;

            let first = read_message(&mut reader)
                .await
                .expect("read first list request");
            assert_eq!(first["method"], "tools/list");
            assert_eq!(first["params"], json!({}));
            write_message(
                &mut writer,
                &json!({
                    "jsonrpc": "2.0",
                    "id": first["id"].clone(),
                    "result": {
                        "tools": [
                            {
                                "name": "alpha",
                                "description": "Alpha",
                                "inputSchema": { "type": "object" }
                            }
                        ],
                        "nextCursor": "cursor-2"
                    }
                }),
            )
            .await
            .expect("write first page");

            let second = read_message(&mut reader)
                .await
                .expect("read second list request");
            assert_eq!(second["method"], "tools/list");
            assert_eq!(second["params"], json!({ "cursor": "cursor-2" }));
            write_message(
                &mut writer,
                &json!({
                    "jsonrpc": "2.0",
                    "id": second["id"].clone(),
                    "result": {
                        "tools": [
                            {
                                "name": "beta",
                                "description": "Beta",
                                "inputSchema": { "type": "object", "properties": { "x": { "type": "string" } } }
                            }
                        ]
                    }
                }),
            )
            .await
            .expect("write second page");
        });

        let mut transport = JsonRpcTransport::new(client_read, client_write);
        let first: McpToolsListResult = serde_json::from_value(
            transport
                .request("tools/list", Some(json!({})))
                .await
                .expect("first page succeeds"),
        )
        .expect("parse first page");
        let second: McpToolsListResult = serde_json::from_value(
            transport
                .request(
                    "tools/list",
                    Some(json!({ "cursor": first.next_cursor.clone().unwrap() })),
                )
                .await
                .expect("second page succeeds"),
        )
        .expect("parse second page");

        let tools = first
            .tools
            .into_iter()
            .chain(second.tools.into_iter())
            .collect::<Vec<_>>();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "alpha");
        assert_eq!(tools[1].name, "beta");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn mcp_tool_adapter_maps_remote_metadata_and_call_output() {
        let client = Arc::new(FakeInvoker::with_output(ToolOutput {
            content: "hello\n{\"type\":\"json\",\"value\":1}".into(),
            structured: Some(json!({ "ok": true })),
        }));
        let adapter = McpToolAdapter::new(
            "deploy".into(),
            McpRemoteTool {
                name: "run".into(),
                description: "Remote deploy".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "service": { "type": "string" }
                    }
                }),
            },
            true,
            client.clone(),
        );

        let spec = adapter.spec();
        assert_eq!(spec.name, "mcp.deploy.run");
        assert_eq!(spec.description, "Remote deploy");
        assert!(spec.requires_approval);
        assert_eq!(adapter.capability_id().kind, CapabilityKind::McpTool);
        assert_eq!(adapter.capability_id().source.as_deref(), Some("deploy"));

        let output = adapter
            .call(
                ToolContext {
                    conversation_id: hajimi_claw_types::ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "service": "api" }),
            )
            .await
            .expect("adapter call succeeds");
        assert_eq!(output.content, "hello\n{\"type\":\"json\",\"value\":1}");
        assert_eq!(output.structured, Some(json!({ "ok": true })));
        assert_eq!(
            client.calls.lock().await[0],
            ("run".to_string(), json!({ "service": "api" }))
        );
    }

    #[test]
    fn flatten_content_blocks_preserves_text_and_json_strings() {
        let content = flatten_content_blocks(&[
            json!({ "type": "text", "text": "hello" }),
            json!({ "type": "json", "value": { "ok": true } }),
            Value::Null,
        ]);
        assert_eq!(
            content,
            "hello\n{\"type\":\"json\",\"value\":{\"ok\":true}}"
        );
    }

    #[tokio::test]
    async fn broken_server_does_not_prevent_healthy_server_registration() {
        let configs = vec![sample_config("broken"), sample_config("healthy")];
        let connector = FakeConnector::new(HashMap::from([
            ("broken".to_string(), Err(String::from("initialize failed"))),
            (
                "healthy".to_string(),
                Ok(FakeConnectedServer {
                    invoker: Arc::new(FakeInvoker::with_output(ToolOutput {
                        content: "ok".into(),
                        structured: None,
                    })),
                    tools: vec![McpRemoteTool {
                        name: "run".into(),
                        description: "Run".into(),
                        input_schema: json!({ "type": "object" }),
                    }],
                }),
            ),
        ]));

        let result = bootstrap_mcp_servers_with_connector(&configs, &connector).await;
        assert_eq!(result.tools.len(), 1);
        assert_eq!(result.statuses.len(), 2);
        assert_eq!(result.statuses[0].name, "broken");
        assert!(!result.statuses[0].connected);
        assert!(result.statuses[0].message.contains("initialize failed"));
        assert_eq!(result.statuses[1].name, "healthy");
        assert!(result.statuses[1].connected);
        assert_eq!(result.statuses[1].tool_count, 1);
    }

    #[tokio::test]
    async fn duplicate_local_names_are_skipped_deterministically() {
        let client = Arc::new(FakeInvoker::with_output(ToolOutput {
            content: "first".into(),
            structured: None,
        }));
        let config = sample_config("deploy");
        let (tools, duplicates) = register_server_tools(
            &config,
            client.clone() as Arc<dyn McpInvoker>,
            vec![
                McpRemoteTool {
                    name: "run".into(),
                    description: "zeta duplicate".into(),
                    input_schema: json!({ "type": "object" }),
                },
                McpRemoteTool {
                    name: "run".into(),
                    description: "alpha duplicate".into(),
                    input_schema: json!({ "type": "object", "properties": { "service": { "type": "string" } } }),
                },
            ],
            &mut Default::default(),
        );
        assert_eq!(duplicates, 1);
        assert_eq!(tools.len(), 1);
        let spec = tools[0].spec();
        assert_eq!(spec.name, "mcp.deploy.run");
        assert_eq!(spec.description, "zeta duplicate");
    }

    fn sample_config(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.into(),
            command: "demo".into(),
            args: vec![],
            cwd: None,
            env_allowlist: vec![],
            startup_timeout_secs: Some(5),
            enabled: true,
            requires_approval: false,
        }
    }

    struct FakeConnectedServer {
        invoker: Arc<FakeInvoker>,
        tools: Vec<McpRemoteTool>,
    }

    struct FakeConnector {
        results: HashMap<String, Result<FakeConnectedServer, String>>,
    }

    impl FakeConnector {
        fn new(results: HashMap<String, Result<FakeConnectedServer, String>>) -> Self {
            Self { results }
        }
    }

    #[async_trait]
    impl McpConnector for FakeConnector {
        async fn connect(&self, config: &McpServerConfig) -> ClawResult<McpConnectedServer> {
            match self.results.get(&config.name) {
                Some(Ok(server)) => Ok(McpConnectedServer {
                    client: server.invoker.clone() as Arc<dyn McpInvoker>,
                    tools: server.tools.clone(),
                }),
                Some(Err(err)) => Err(ClawError::Backend(err.clone())),
                None => Err(ClawError::Backend(format!(
                    "missing fake connector result for {}",
                    config.name
                ))),
            }
        }
    }

    struct FakeInvoker {
        output: ToolOutput,
        calls: Mutex<Vec<(String, Value)>>,
    }

    impl FakeInvoker {
        fn with_output(output: ToolOutput) -> Self {
            Self {
                output,
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl McpInvoker for FakeInvoker {
        async fn call_tool(&self, tool_name: &str, input: Value) -> ClawResult<ToolOutput> {
            self.calls.lock().await.push((tool_name.to_string(), input));
            Ok(self.output.clone())
        }
    }
}
