#[tokio::main]
async fn main() -> anyhow::Result<()> {
    hajimi_claw_daemon::entry_from_env().await
}
