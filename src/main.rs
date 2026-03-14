#[tokio::main]
async fn main() -> anyhow::Result<()> {
    hajimi_claw_daemon::run_from_env().await
}
