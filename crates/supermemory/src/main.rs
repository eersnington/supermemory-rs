#[tokio::main]
async fn main() -> Result<(), supermemory::StartupError> {
    supermemory::run().await
}
