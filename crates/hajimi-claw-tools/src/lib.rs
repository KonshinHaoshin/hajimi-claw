use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use hajimi_claw_policy::PolicyEngine;
use hajimi_claw_types::{
    ClawError, ClawResult, ExecRequest, Executor, SessionId, SessionOpenRequest, Tool, ToolContext,
    ToolOutput, ToolSpec,
};
use regex::Regex;
use reqwest::Method;
use rustls::pki_types::ServerName;
use rustls::{ClientConfig, RootCertStore};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::fs;
use tokio::net::{TcpStream, lookup_host};
use tokio::time::{Duration, timeout};
use tokio_rustls::TlsConnector;
use url::Url;

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        let tools = tools
            .into_iter()
            .map(|tool| (tool.spec().name.clone(), tool))
            .collect();
        Self { tools }
    }

    pub fn default(executor: Arc<dyn Executor>, policy: Arc<PolicyEngine>) -> Self {
        Self::new(vec![
            Arc::new(ReadFileTool::new(policy.clone())),
            Arc::new(TailFileTool::new(policy.clone())),
            Arc::new(ListDirTool::new(policy.clone())),
            Arc::new(GrepTextTool::new(policy.clone())),
            Arc::new(WriteFileTool::new(policy.clone())),
            Arc::new(AppendFileTool::new(policy.clone())),
            Arc::new(HttpProbeTool::new()),
            Arc::new(CurlRequestTool::new()),
            Arc::new(DnsLookupTool::new()),
            Arc::new(PortCheckTool::new()),
            Arc::new(TlsCheckTool::new()),
            Arc::new(SystemdStatusTool::new(executor.clone())),
            Arc::new(SystemdRestartTool::new(executor.clone())),
            Arc::new(DockerPsTool::new(executor.clone())),
            Arc::new(DockerLogsTool::new(executor.clone())),
            Arc::new(DockerRestartTool::new(executor.clone())),
            Arc::new(RunCommandTool::new(executor.clone())),
            Arc::new(ExecOnceTool::new(executor.clone())),
            Arc::new(PingHostTool::new(executor.clone())),
            Arc::new(SessionOpenTool::new(executor.clone())),
            Arc::new(SessionExecTool::new(executor.clone())),
            Arc::new(SessionCloseTool::new(executor)),
        ])
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|tool| tool.spec()).collect()
    }

    pub async fn call(&self, name: &str, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ClawError::NotFound(format!("tool not found: {name}")))?;
        tool.call(ctx, input).await
    }
}

fn object_schema(properties: Vec<(&str, Value)>, required: &[&str]) -> Value {
    let mut map = serde_json::Map::new();
    for (name, schema) in properties {
        map.insert(name.to_string(), schema);
    }
    json!({
        "type": "object",
        "properties": map,
        "required": required,
        "additionalProperties": false,
    })
}

fn string_schema(description: &str) -> Value {
    json!({
        "type": "string",
        "description": description,
    })
}

fn integer_schema(description: &str, minimum: Option<u64>) -> Value {
    let mut schema = json!({
        "type": "integer",
        "description": description,
    });
    if let Some(minimum) = minimum {
        schema["minimum"] = json!(minimum);
    }
    schema
}

fn bool_schema(description: &str) -> Value {
    json!({
        "type": "boolean",
        "description": description,
    })
}

fn string_array_schema(description: &str) -> Value {
    json!({
        "type": "array",
        "description": description,
        "items": {
            "type": "string",
        },
    })
}

fn string_map_schema(description: &str) -> Value {
    json!({
        "type": "object",
        "description": description,
        "additionalProperties": {
            "type": "string",
        },
    })
}

#[derive(Debug, Clone)]
struct HttpRequestOptions {
    url: String,
    method: Method,
    headers: HashMap<String, String>,
    body: Option<String>,
    max_body_bytes: usize,
    follow_redirects: bool,
}

async fn perform_http_request(input: HttpRequestOptions) -> ClawResult<ToolOutput> {
    let client = reqwest::Client::builder()
        .redirect(if input.follow_redirects {
            reqwest::redirect::Policy::limited(5)
        } else {
            reqwest::redirect::Policy::none()
        })
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| ClawError::Backend(err.to_string()))?;
    let mut request = client.request(input.method.clone(), &input.url);
    for (key, value) in input.headers {
        request = request.header(&key, value);
    }
    if let Some(body) = input.body {
        request = request.body(body);
    }

    let started = Instant::now();
    let response = request
        .send()
        .await
        .map_err(|err| ClawError::Backend(err.to_string()))?;
    let elapsed_ms = started.elapsed().as_millis();
    let status = response.status();
    let final_url = response.url().to_string();
    let headers = response
        .headers()
        .iter()
        .map(|(key, value)| {
            (
                key.to_string(),
                value.to_str().unwrap_or("<non-utf8>").to_string(),
            )
        })
        .collect::<HashMap<_, _>>();
    let body = response
        .text()
        .await
        .map_err(|err| ClawError::Backend(err.to_string()))?;
    let body_sample = truncate_string(body, input.max_body_bytes);

    Ok(ToolOutput {
        content: format!(
            "status={}\nfinal_url={}\nduration_ms={}\nbody:\n{}",
            status.as_u16(),
            final_url,
            elapsed_ms,
            body_sample
        ),
        structured: Some(json!({
            "status": status.as_u16(),
            "final_url": final_url,
            "duration_ms": elapsed_ms,
            "headers": headers,
            "body_sample": body_sample,
            "method": input.method.as_str(),
        })),
    })
}

struct ReadFileTool {
    policy: Arc<PolicyEngine>,
}

impl ReadFileTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read a text file from an allowed directory.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("path", string_schema("Path to the file to read.")),
                    (
                        "max_bytes",
                        integer_schema("Maximum number of bytes to return.", Some(1)),
                    ),
                ],
                &["path"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            max_bytes: Option<usize>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_path(&self.policy, &input.path)?;
        let content = fs::read_to_string(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let max_bytes = input.max_bytes.unwrap_or(8 * 1024);
        let content = truncate_string(content, max_bytes);
        Ok(ToolOutput {
            content,
            structured: None,
        })
    }
}

struct TailFileTool {
    policy: Arc<PolicyEngine>,
}

impl TailFileTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for TailFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tail_file".into(),
            description: "Read the tail of a text file.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("path", string_schema("Path to the file to tail.")),
                    (
                        "lines",
                        integer_schema("Number of lines to return.", Some(1)),
                    ),
                ],
                &["path"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            lines: Option<usize>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_path(&self.policy, &input.path)?;
        let content = fs::read_to_string(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let lines = input.lines.unwrap_or(50);
        let tail = content
            .lines()
            .rev()
            .take(lines)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolOutput {
            content: tail,
            structured: None,
        })
    }
}

struct ListDirTool {
    policy: Arc<PolicyEngine>,
}

impl ListDirTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".into(),
            description: "List files and folders in an allowed directory.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![("path", string_schema("Directory path to list."))],
                &["path"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_dir(&self.policy, &input.path)?;
        let mut dir = fs::read_dir(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let mut entries = Vec::new();
        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?
        {
            let metadata = entry
                .metadata()
                .await
                .map_err(|err| ClawError::Backend(err.to_string()))?;
            entries.push(format!(
                "{}\t{}",
                if metadata.is_dir() { "dir" } else { "file" },
                entry.path().display()
            ));
        }
        Ok(ToolOutput {
            content: entries.join("\n"),
            structured: None,
        })
    }
}

struct GrepTextTool {
    policy: Arc<PolicyEngine>,
}

impl GrepTextTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

struct WriteFileTool {
    policy: Arc<PolicyEngine>,
}

impl WriteFileTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description:
                "Write text to a file in a writable allowed directory, replacing existing content."
                    .into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("path", string_schema("Path to write.")),
                    ("content", string_schema("Full replacement content.")),
                ],
                &["path", "content"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            content: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_write_path(&self.policy, &input.path)?;
        if let Some(parent) = input.path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|err| ClawError::Backend(err.to_string()))?;
        }
        fs::write(&input.path, input.content.as_bytes())
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        Ok(ToolOutput {
            content: format!("wrote {}", input.path.display()),
            structured: None,
        })
    }
}

struct AppendFileTool {
    policy: Arc<PolicyEngine>,
}

impl AppendFileTool {
    fn new(policy: Arc<PolicyEngine>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Tool for AppendFileTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "append_file".into(),
            description: "Append text to a file in a writable allowed directory.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("path", string_schema("Path to append to.")),
                    ("content", string_schema("Text to append.")),
                ],
                &["path", "content"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            content: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_write_path(&self.policy, &input.path)?;
        if let Some(parent) = input.path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|err| ClawError::Backend(err.to_string()))?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        use tokio::io::AsyncWriteExt;
        file.write_all(input.content.as_bytes())
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        Ok(ToolOutput {
            content: format!("appended {}", input.path.display()),
            structured: None,
        })
    }
}

#[async_trait]
impl Tool for GrepTextTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep_text".into(),
            description: "Search a text file with a regex pattern.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("path", string_schema("Path to the file to search.")),
                    ("pattern", string_schema("Regex pattern to search for.")),
                ],
                &["path", "pattern"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            path: PathBuf,
            pattern: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        guard_path(&self.policy, &input.path)?;
        let pattern =
            Regex::new(&input.pattern).map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let content = fs::read_to_string(&input.path)
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?;
        let matches = content
            .lines()
            .enumerate()
            .filter_map(|(index, line)| {
                pattern
                    .is_match(line)
                    .then(|| format!("{}:{}", index + 1, line))
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolOutput {
            content: matches,
            structured: None,
        })
    }
}

struct HttpProbeTool;

impl HttpProbeTool {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for HttpProbeTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "http_probe".into(),
            description: "Make an HTTP request and return status, timing, and a short body sample."
                .into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("url", string_schema("Absolute HTTP or HTTPS URL to probe.")),
                    ("method", string_schema("HTTP method, default GET.")),
                    ("headers", string_map_schema("Optional HTTP headers.")),
                    (
                        "body",
                        string_schema("Optional request body for POST-like methods."),
                    ),
                    (
                        "follow_redirects",
                        bool_schema("Whether to follow up to 5 redirects. Default true."),
                    ),
                    (
                        "max_body_bytes",
                        integer_schema(
                            "Maximum body bytes to include in the response sample.",
                            Some(1),
                        ),
                    ),
                ],
                &["url"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            url: String,
            method: Option<String>,
            headers: Option<HashMap<String, String>>,
            body: Option<String>,
            follow_redirects: Option<bool>,
            max_body_bytes: Option<usize>,
        }

        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let method = input
            .method
            .as_deref()
            .unwrap_or("GET")
            .parse::<Method>()
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        perform_http_request(HttpRequestOptions {
            url: input.url,
            method,
            headers: input.headers.unwrap_or_default(),
            body: input.body,
            max_body_bytes: input.max_body_bytes.unwrap_or(2048),
            follow_redirects: input.follow_redirects.unwrap_or(true),
        })
        .await
    }
}

struct CurlRequestTool;

impl CurlRequestTool {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for CurlRequestTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "curl_request".into(),
            description:
                "Send an HTTP request similar to curl, including headers, redirects, and body."
                    .into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    (
                        "url",
                        string_schema("Absolute HTTP or HTTPS URL to request."),
                    ),
                    ("method", string_schema("HTTP method, default GET.")),
                    ("headers", string_map_schema("Optional request headers.")),
                    ("body", string_schema("Optional request body.")),
                    (
                        "follow_redirects",
                        bool_schema("Whether to follow redirects. Default true."),
                    ),
                    (
                        "max_body_bytes",
                        integer_schema(
                            "Maximum body bytes to include in the response sample.",
                            Some(1),
                        ),
                    ),
                ],
                &["url"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            url: String,
            method: Option<String>,
            headers: Option<HashMap<String, String>>,
            body: Option<String>,
            follow_redirects: Option<bool>,
            max_body_bytes: Option<usize>,
        }

        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let method = input
            .method
            .as_deref()
            .unwrap_or("GET")
            .parse::<Method>()
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        perform_http_request(HttpRequestOptions {
            url: input.url,
            method,
            headers: input.headers.unwrap_or_default(),
            body: input.body,
            max_body_bytes: input.max_body_bytes.unwrap_or(4096),
            follow_redirects: input.follow_redirects.unwrap_or(true),
        })
        .await
    }
}

struct DnsLookupTool;

impl DnsLookupTool {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for DnsLookupTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "dns_lookup".into(),
            description: "Resolve a hostname to IP addresses using the local resolver.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("host", string_schema("Hostname to resolve.")),
                    (
                        "port",
                        integer_schema("Port to pair with the lookup, default 0.", Some(0)),
                    ),
                ],
                &["host"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            host: String,
            port: Option<u16>,
        }

        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let addresses = lookup_host((input.host.as_str(), input.port.unwrap_or(0)))
            .await
            .map_err(|err| ClawError::Backend(err.to_string()))?
            .map(|addr| addr.ip().to_string())
            .collect::<Vec<_>>();

        let mut unique = Vec::new();
        for address in addresses {
            if !unique.iter().any(|item| item == &address) {
                unique.push(address);
            }
        }

        Ok(ToolOutput {
            content: if unique.is_empty() {
                format!("no addresses resolved for {}", input.host)
            } else {
                unique.join("\n")
            },
            structured: Some(json!({
                "host": input.host,
                "addresses": unique,
            })),
        })
    }
}

struct PortCheckTool;

impl PortCheckTool {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for PortCheckTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "port_check".into(),
            description: "Check whether a TCP port is reachable from this host.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("host", string_schema("Target hostname or IP.")),
                    ("port", integer_schema("TCP port to check.", Some(1))),
                    (
                        "timeout_secs",
                        integer_schema("Connection timeout in seconds.", Some(1)),
                    ),
                ],
                &["host", "port"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            host: String,
            port: u16,
            timeout_secs: Option<u64>,
        }

        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let timeout_secs = input.timeout_secs.unwrap_or(5).max(1);
        let address = format!("{}:{}", input.host, input.port);
        let started = Instant::now();
        let result = timeout(
            Duration::from_secs(timeout_secs),
            TcpStream::connect(&address),
        )
        .await;

        let (open, message) = match result {
            Ok(Ok(_stream)) => (true, "connected".to_string()),
            Ok(Err(err)) => (false, err.to_string()),
            Err(_) => (false, format!("timed out after {timeout_secs}s")),
        };

        Ok(ToolOutput {
            content: format!(
                "host={}\nport={}\nopen={}\nduration_ms={}\nmessage={}",
                input.host,
                input.port,
                open,
                started.elapsed().as_millis(),
                message
            ),
            structured: Some(json!({
                "host": input.host,
                "port": input.port,
                "open": open,
                "duration_ms": started.elapsed().as_millis(),
                "message": message,
            })),
        })
    }
}

struct TlsCheckTool;

impl TlsCheckTool {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for TlsCheckTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tls_check".into(),
            description: "Open a TLS connection and report negotiated protocol details.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("url", string_schema("HTTPS URL to inspect.")),
                    ("host", string_schema("Optional hostname override.")),
                    (
                        "port",
                        integer_schema("Optional TCP port override.", Some(1)),
                    ),
                    (
                        "timeout_secs",
                        integer_schema("Connection timeout in seconds.", Some(1)),
                    ),
                ],
                &["url"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            url: String,
            host: Option<String>,
            port: Option<u16>,
            timeout_secs: Option<u64>,
        }

        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let parsed = Url::parse(&input.url)
            .map_err(|err| ClawError::InvalidRequest(format!("invalid url: {err}")))?;
        let host = input
            .host
            .or_else(|| parsed.host_str().map(ToString::to_string))
            .ok_or_else(|| ClawError::InvalidRequest("url does not contain a hostname".into()))?;
        let port = input.port.or_else(|| parsed.port()).unwrap_or(443);
        let timeout_secs = input.timeout_secs.unwrap_or(10).max(1);

        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = TlsConnector::from(Arc::new(config));
        let server_name = ServerName::try_from(host.clone())
            .map_err(|_| ClawError::InvalidRequest(format!("invalid tls server name: {host}")))?;

        let started = Instant::now();
        let tcp = timeout(
            Duration::from_secs(timeout_secs),
            TcpStream::connect((host.as_str(), port)),
        )
        .await
        .map_err(|_| ClawError::Backend(format!("tcp connect timed out after {timeout_secs}s")))?
        .map_err(|err| ClawError::Backend(err.to_string()))?;
        let tls = timeout(
            Duration::from_secs(timeout_secs),
            connector.connect(server_name, tcp),
        )
        .await
        .map_err(|_| ClawError::Backend(format!("tls handshake timed out after {timeout_secs}s")))?
        .map_err(|err| ClawError::Backend(err.to_string()))?;
        let elapsed_ms = started.elapsed().as_millis();
        let (_, session) = tls.get_ref();
        let protocol = session
            .protocol_version()
            .map(|value| format!("{value:?}"))
            .unwrap_or_else(|| "unknown".into());
        let alpn = session
            .alpn_protocol()
            .map(|value| String::from_utf8_lossy(value).to_string());
        let certificate_count = session
            .peer_certificates()
            .map(|certs| certs.len())
            .unwrap_or_default();
        let cipher_suite = session
            .negotiated_cipher_suite()
            .map(|suite| format!("{:?}", suite.suite()))
            .unwrap_or_else(|| "unknown".into());

        Ok(ToolOutput {
            content: format!(
                "host={}\nport={}\nprotocol={}\ncipher_suite={}\nalpn={}\npeer_certificates={}\nduration_ms={}",
                host,
                port,
                protocol,
                cipher_suite,
                alpn.clone().unwrap_or_else(|| "none".into()),
                certificate_count,
                elapsed_ms
            ),
            structured: Some(json!({
                "host": host,
                "port": port,
                "protocol": protocol,
                "cipher_suite": cipher_suite,
                "alpn": alpn,
                "peer_certificates": certificate_count,
                "duration_ms": elapsed_ms,
            })),
        })
    }
}

struct SystemdStatusTool {
    executor: Arc<dyn Executor>,
}

impl SystemdStatusTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SystemdStatusTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "systemd_status".into(),
            description: "Inspect a systemd service status.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![("service", string_schema("Systemd service name."))],
                &["service"],
            ),
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            service: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "systemctl".into(),
                args: vec!["status".into(), "--no-pager".into(), input.service],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 30,
                max_output_bytes: 16 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct SystemdRestartTool {
    executor: Arc<dyn Executor>,
}

impl SystemdRestartTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SystemdRestartTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "systemd_restart".into(),
            description: "Restart a systemd service.".into(),
            requires_approval: true,
            input_schema: object_schema(
                vec![("service", string_schema("Systemd service name."))],
                &["service"],
            ),
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            service: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "systemctl".into(),
                args: vec!["restart".into(), input.service],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 30,
                max_output_bytes: 8 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct DockerPsTool {
    executor: Arc<dyn Executor>,
}

impl DockerPsTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for DockerPsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "docker_ps".into(),
            description: "List running containers.".into(),
            requires_approval: false,
            input_schema: object_schema(vec![], &[]),
        }
    }

    async fn call(&self, ctx: ToolContext, _input: Value) -> ClawResult<ToolOutput> {
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "docker".into(),
                args: vec!["ps".into()],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 20,
                max_output_bytes: 12 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct DockerLogsTool {
    executor: Arc<dyn Executor>,
}

impl DockerLogsTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for DockerLogsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "docker_logs".into(),
            description: "Fetch recent logs from a container.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("container", string_schema("Container name or ID.")),
                    (
                        "tail",
                        integer_schema("How many recent lines to fetch.", Some(1)),
                    ),
                ],
                &["container"],
            ),
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            container: String,
            tail: Option<u32>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "docker".into(),
                args: vec![
                    "logs".into(),
                    "--tail".into(),
                    input.tail.unwrap_or(200).to_string(),
                    input.container,
                ],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 30,
                max_output_bytes: 16 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct DockerRestartTool {
    executor: Arc<dyn Executor>,
}

impl DockerRestartTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for DockerRestartTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "docker_restart".into(),
            description: "Restart a Docker container.".into(),
            requires_approval: true,
            input_schema: object_schema(
                vec![("container", string_schema("Container name or ID."))],
                &["container"],
            ),
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            container: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "docker".into(),
                args: vec!["restart".into(), input.container],
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: 30,
                max_output_bytes: 8 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct RunCommandTool {
    executor: Arc<dyn Executor>,
}

impl RunCommandTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for RunCommandTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_command".into(),
            description: "Run a guarded command once.".into(),
            requires_approval: true,
            input_schema: object_schema(
                vec![
                    ("command", string_schema("Executable name or path.")),
                    ("args", string_array_schema("Command arguments.")),
                    ("cwd", string_schema("Working directory override.")),
                ],
                &["command"],
            ),
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            command: String,
            args: Option<Vec<String>>,
            cwd: Option<PathBuf>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: input.command,
                args: input.args.unwrap_or_default(),
                cwd: input.cwd.or(ctx.working_directory),
                env_allowlist: vec![],
                timeout_secs: 60,
                max_output_bytes: 24 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct ExecOnceTool {
    executor: Arc<dyn Executor>,
}

impl ExecOnceTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for ExecOnceTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "exec_once".into(),
            description: "Run one guarded terminal command with explicit argv.".into(),
            requires_approval: true,
            input_schema: object_schema(
                vec![
                    ("command", string_schema("Executable name or path.")),
                    ("args", string_array_schema("Explicit argument vector.")),
                    ("cwd", string_schema("Working directory override.")),
                    (
                        "timeout_secs",
                        integer_schema("Execution timeout in seconds.", Some(1)),
                    ),
                    (
                        "max_output_bytes",
                        integer_schema("Maximum stdout/stderr bytes to retain.", Some(256)),
                    ),
                    (
                        "requires_tty",
                        bool_schema("Whether the command requires a TTY-like environment."),
                    ),
                ],
                &["command"],
            ),
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            command: String,
            args: Option<Vec<String>>,
            cwd: Option<PathBuf>,
            timeout_secs: Option<u64>,
            max_output_bytes: Option<usize>,
            requires_tty: Option<bool>,
        }

        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let result = self
            .executor
            .run_once(ExecRequest {
                command: input.command,
                args: input.args.unwrap_or_default(),
                cwd: input.cwd.or(ctx.working_directory),
                env_allowlist: vec![],
                timeout_secs: input.timeout_secs.unwrap_or(60).max(1),
                max_output_bytes: input.max_output_bytes.unwrap_or(24 * 1024).max(256),
                requires_tty: input.requires_tty.unwrap_or(false),
            })
            .await?;
        Ok(command_output(result))
    }
}

struct PingHostTool {
    executor: Arc<dyn Executor>,
}

impl PingHostTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for PingHostTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "ping_host".into(),
            description: "Ping a host using the local system ping command.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("host", string_schema("Hostname or IP address to ping.")),
                    (
                        "count",
                        integer_schema("Number of echo requests to send.", Some(1)),
                    ),
                    (
                        "timeout_secs",
                        integer_schema("Timeout in seconds for each attempt.", Some(1)),
                    ),
                ],
                &["host"],
            ),
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            host: String,
            count: Option<u32>,
            timeout_secs: Option<u64>,
        }

        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let count = input.count.unwrap_or(4).max(1);
        let timeout_secs = input.timeout_secs.unwrap_or(5).max(1);
        let args = if cfg!(windows) {
            vec![
                "-n".into(),
                count.to_string(),
                "-w".into(),
                (timeout_secs * 1000).to_string(),
                input.host,
            ]
        } else {
            vec![
                "-n".into(),
                "-c".into(),
                count.to_string(),
                "-W".into(),
                timeout_secs.to_string(),
                input.host,
            ]
        };
        let result = self
            .executor
            .run_once(ExecRequest {
                command: "ping".into(),
                args,
                cwd: ctx.working_directory,
                env_allowlist: vec![],
                timeout_secs: (timeout_secs * count as u64).max(timeout_secs),
                max_output_bytes: 16 * 1024,
                requires_tty: false,
            })
            .await?;
        Ok(command_output(result))
    }
}

struct SessionOpenTool {
    executor: Arc<dyn Executor>,
}

impl SessionOpenTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SessionOpenTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "session_open".into(),
            description: "Open a persistent shell session.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![
                    ("name", string_schema("Optional session name.")),
                    ("cwd", string_schema("Optional working directory.")),
                ],
                &[],
            ),
        }
    }

    async fn call(&self, ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            name: Option<String>,
            cwd: Option<PathBuf>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let session = self
            .executor
            .open_session(SessionOpenRequest {
                name: input.name,
                cwd: input.cwd.or(ctx.working_directory),
                env_allowlist: vec![],
            })
            .await?;
        Ok(ToolOutput {
            content: format!("opened session {} at {}", session.id, session.cwd.display()),
            structured: Some(json!({
                "session_id": session.id.to_string(),
                "cwd": session.cwd,
                "name": session.name,
            })),
        })
    }
}

struct SessionExecTool {
    executor: Arc<dyn Executor>,
}

impl SessionExecTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SessionExecTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "session_exec".into(),
            description: "Run a command in an existing session.".into(),
            requires_approval: true,
            input_schema: object_schema(
                vec![
                    ("session_id", string_schema("Session identifier.")),
                    (
                        "command",
                        string_schema("Shell command to execute in the session."),
                    ),
                ],
                &["session_id", "command"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            session_id: String,
            command: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let session_id = SessionId(
            uuid::Uuid::parse_str(&input.session_id)
                .map_err(|err| ClawError::InvalidRequest(err.to_string()))?,
        );
        let (command, args) = shell_request_parts(&input.command);
        let result = self
            .executor
            .run_in_session(
                session_id,
                ExecRequest {
                    command,
                    args,
                    cwd: None,
                    env_allowlist: vec![],
                    timeout_secs: 120,
                    max_output_bytes: 24 * 1024,
                    requires_tty: false,
                },
            )
            .await?;
        Ok(command_output(result))
    }
}

struct SessionCloseTool {
    executor: Arc<dyn Executor>,
}

impl SessionCloseTool {
    fn new(executor: Arc<dyn Executor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Tool for SessionCloseTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "session_close".into(),
            description: "Close a shell session.".into(),
            requires_approval: false,
            input_schema: object_schema(
                vec![("session_id", string_schema("Session identifier."))],
                &["session_id"],
            ),
        }
    }

    async fn call(&self, _ctx: ToolContext, input: Value) -> ClawResult<ToolOutput> {
        #[derive(Deserialize)]
        struct Input {
            session_id: String,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|err| ClawError::InvalidRequest(err.to_string()))?;
        let session_id = SessionId(
            uuid::Uuid::parse_str(&input.session_id)
                .map_err(|err| ClawError::InvalidRequest(err.to_string()))?,
        );
        self.executor.close_session(session_id).await?;
        Ok(ToolOutput {
            content: format!("closed session {}", input.session_id),
            structured: None,
        })
    }
}

fn shell_request_parts(command: &str) -> (String, Vec<String>) {
    if cfg!(windows) {
        ("cmd".into(), vec!["/C".into(), command.into()])
    } else {
        ("sh".into(), vec!["-lc".into(), command.into()])
    }
}

fn command_output(result: hajimi_claw_types::ExecResult) -> ToolOutput {
    ToolOutput {
        content: format_output(
            &result.stdout,
            &result.stderr,
            result.exit_code,
            result.truncated,
        ),
        structured: Some(json!({
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "truncated": result.truncated,
            "duration_ms": result.duration_ms,
        })),
    }
}

fn format_output(stdout: &str, stderr: &str, exit_code: Option<i32>, truncated: bool) -> String {
    let mut parts = vec![format!("exit_code={}", exit_code.unwrap_or(-1))];
    if !stdout.trim().is_empty() {
        parts.push(format!("stdout:\n{}", stdout.trim()));
    }
    if !stderr.trim().is_empty() {
        parts.push(format!("stderr:\n{}", stderr.trim()));
    }
    if truncated {
        parts.push("output was truncated".into());
    }
    parts.join("\n\n")
}

fn guard_path(policy: &PolicyEngine, path: &Path) -> ClawResult<()> {
    let candidate = path.parent().unwrap_or(path);
    guard_dir(policy, candidate)
}

fn guard_dir(policy: &PolicyEngine, path: &Path) -> ClawResult<()> {
    if !policy.is_allowed_workdir(path) {
        return Err(ClawError::AccessDenied(format!(
            "path is outside allowed directories: {}",
            path.display()
        )));
    }
    Ok(())
}

fn guard_write_path(policy: &PolicyEngine, path: &Path) -> ClawResult<()> {
    let candidate = path.parent().unwrap_or(path);
    guard_dir(policy, candidate)?;
    if !policy.is_writable_workdir(candidate) {
        return Err(ClawError::AccessDenied(format!(
            "path is outside writable directories: {}",
            candidate.display()
        )));
    }
    Ok(())
}

fn truncate_string(content: String, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content;
    }
    let tail = &content[content.len() - max_bytes..];
    format!("...[truncated]\n{tail}")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use hajimi_claw_exec::{LocalExecutor, PlatformMode};
    use hajimi_claw_policy::{PolicyConfig, PolicyEngine};
    use hajimi_claw_types::{
        ClawError, ConversationId, ExecRequest, ExecResult, Executor, SessionHandle, SessionId,
        SessionOpenRequest, ToolContext,
    };
    use serde_json::json;
    use tempfile::tempdir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;

    use super::ToolRegistry;

    #[tokio::test]
    async fn read_file_respects_policy() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("demo.txt");
        tokio::fs::write(&path, "hello").await?;

        let mut config = PolicyConfig::default();
        config.allowed_workdirs = vec![dir.path().to_path_buf()];
        let policy = Arc::new(PolicyEngine::new(config));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = ToolRegistry::default(executor, policy);

        let output = tools
            .call(
                "read_file",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "path": path }),
            )
            .await?;
        assert_eq!(output.content, "hello");
        Ok(())
    }

    #[tokio::test]
    async fn http_probe_returns_status_and_body() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer).await.expect("read request");
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nContent-Type: text/plain\r\n\r\nhello",
                )
                .await
                .expect("write response");
        });

        let policy = Arc::new(PolicyEngine::new(PolicyConfig::default()));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = ToolRegistry::default(executor, policy);
        let output = tools
            .call(
                "http_probe",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "url": format!("http://{addr}/") }),
            )
            .await?;

        server.await?;
        assert!(output.content.contains("status=200"));
        assert!(output.content.contains("hello"));
        Ok(())
    }

    #[tokio::test]
    async fn port_check_reports_open_listener() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let accept_task = tokio::spawn(async move {
            let _ = listener.accept().await;
        });

        let policy = Arc::new(PolicyEngine::new(PolicyConfig::default()));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = ToolRegistry::default(executor, policy);
        let output = tools
            .call(
                "port_check",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "host": "127.0.0.1", "port": addr.port() }),
            )
            .await?;

        accept_task.await?;
        assert!(output.content.contains("open=true"));
        Ok(())
    }

    #[tokio::test]
    async fn dns_lookup_resolves_localhost() -> Result<()> {
        let policy = Arc::new(PolicyEngine::new(PolicyConfig::default()));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = ToolRegistry::default(executor, policy);
        let output = tools
            .call(
                "dns_lookup",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "host": "localhost" }),
            )
            .await?;

        assert!(
            output.content.contains("127.0.0.1") || output.content.contains("::1"),
            "unexpected dns output: {}",
            output.content
        );
        Ok(())
    }

    #[tokio::test]
    async fn curl_request_returns_response_body() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer).await.expect("read request");
            stream
                .write_all(
                    b"HTTP/1.1 201 Created\r\nContent-Length: 7\r\nContent-Type: text/plain\r\n\r\ncreated",
                )
                .await
                .expect("write response");
        });

        let policy = Arc::new(PolicyEngine::new(PolicyConfig::default()));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = ToolRegistry::default(executor, policy);
        let output = tools
            .call(
                "curl_request",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "url": format!("http://{addr}/"), "method": "POST", "body": "demo" }),
            )
            .await?;

        server.await?;
        assert!(output.content.contains("status=201"));
        assert!(output.content.contains("created"));
        Ok(())
    }

    struct FakeExecutor {
        last_request: Mutex<Option<ExecRequest>>,
    }

    #[async_trait::async_trait]
    impl Executor for FakeExecutor {
        async fn run_once(&self, req: ExecRequest) -> Result<ExecResult, ClawError> {
            *self.last_request.lock().await = Some(req);
            Ok(ExecResult {
                exit_code: Some(0),
                stdout: "pong".into(),
                stderr: String::new(),
                duration_ms: 5,
                truncated: false,
            })
        }

        async fn open_session(&self, _req: SessionOpenRequest) -> Result<SessionHandle, ClawError> {
            Err(ClawError::Backend("not used".into()))
        }

        async fn run_in_session(
            &self,
            _id: SessionId,
            _req: ExecRequest,
        ) -> Result<ExecResult, ClawError> {
            Err(ClawError::Backend("not used".into()))
        }

        async fn close_session(&self, _id: SessionId) -> Result<(), ClawError> {
            Err(ClawError::Backend("not used".into()))
        }
    }

    #[tokio::test]
    async fn ping_host_builds_ping_command() -> Result<()> {
        let executor = Arc::new(FakeExecutor {
            last_request: Mutex::new(None),
        });
        let policy = Arc::new(PolicyEngine::new(PolicyConfig::default()));
        let tools = ToolRegistry::default(executor.clone(), policy);
        let output = tools
            .call(
                "ping_host",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "host": "example.com", "count": 2, "timeout_secs": 3 }),
            )
            .await?;

        let request = executor
            .last_request
            .lock()
            .await
            .clone()
            .expect("captured request");
        assert_eq!(request.command, "ping");
        assert!(request.args.iter().any(|arg| arg == "example.com"));
        assert!(output.content.contains("exit_code=0"));
        Ok(())
    }

    #[tokio::test]
    async fn tls_check_rejects_invalid_url() -> Result<()> {
        let policy = Arc::new(PolicyEngine::new(PolicyConfig::default()));
        let executor = Arc::new(LocalExecutor::new(
            policy.clone(),
            PlatformMode::WindowsSafe,
        ));
        let tools = ToolRegistry::default(executor, policy);
        let err = tools
            .call(
                "tls_check",
                ToolContext {
                    conversation_id: ConversationId::new(),
                    working_directory: None,
                    elevated: false,
                },
                json!({ "url": "not-a-url" }),
            )
            .await
            .expect_err("invalid tls input should fail");

        assert!(matches!(err, ClawError::InvalidRequest(_)));
        Ok(())
    }
}
