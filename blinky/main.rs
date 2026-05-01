use bssl_rustls_adapters::CryptoProviderBuilder;
use dotenvy::dotenv;
use rig::{
    agent::Agent,
    client::CompletionClient,
    completion::Chat as _,
    message::Message,
    providers::{gemini, openai},
};
use rootcause::{Report, bail, option_ext::OptionExt as _, prelude::ResultExt as _};
use rustls::client::ClientConfig;
use rustls::crypto::CryptoProvider;
use std::{
    collections::HashMap,
    env,
    sync::{Arc, RwLock},
};
use tokio::sync::Mutex;
use tracing::Instrument;
use twilight_cache_inmemory::{DefaultInMemoryCache, ResourceType};
use twilight_gateway::{Event, EventTypeFlags, Intents, Shard, ShardId, StreamExt as _};
use twilight_http::Client as HttpClient;
use twilight_model::id::{Id, marker::UserMarker};

type UserHistories = Arc<RwLock<HashMap<String, Vec<Message>>>>;

pub struct State {
    bot_id: Id<UserMarker>,
    openai_agent: Arc<Agent<openai::CompletionModel>>,
    gemini_agent: Arc<Agent<gemini::CompletionModel>>,
    histories: UserHistories,
}

#[tokio::main]
async fn main() -> Result<(), Report> {
    dotenv().context("Failed to load .env file")?;
    tracing_subscriber::fmt::init();

    let provider = CryptoProviderBuilder::full();
    provider
        .install_default()
        .expect("Failed to install default crypto provider");

    let token = env::var("DISCORD_TOKEN")?;

    let intents = Intents::GUILD_MESSAGES | Intents::DIRECT_MESSAGES | Intents::MESSAGE_CONTENT;

    let mut shard = Shard::new(ShardId::ONE, token.clone(), intents);

    let http = Arc::new(HttpClient::new(token));

    let cache = DefaultInMemoryCache::builder()
        .resource_types(ResourceType::MESSAGE)
        .build();

    let state = Arc::new(Mutex::new(None));

    while let Some(item) = shard.next_event(EventTypeFlags::all()).await {
        let state = Arc::clone(&state);
        let http = Arc::clone(&http);

        let Ok(event) = item else {
            tracing::warn!(source = ?item.unwrap_err(), "error receiving event");

            continue;
        };
        cache.update(&event);

        tokio::spawn(
            async move {
                if let Err(e) = handle_event(event, Arc::clone(&http), Arc::clone(&state)).await {
                    tracing::warn!(source = ?e, "error handling event");
                }
            }
            .instrument(tracing::Span::current()),
        );
    }

    Ok(())
}

async fn handle_event(
    event: Event,
    http: Arc<HttpClient>,
    state: Arc<Mutex<Option<State>>>,
) -> Result<(), Report> {
    if let Event::Ready(ready) = event {
        let openai_agent = Arc::new(
            openai::Client::builder()
                .base_url(
                    env::var("OPENAI_BASE_URL")
                        .unwrap_or_else(|_| "https://api.openai.com".to_string()),
                )
                .api_key(env::var("OPENAI_API_KEY")?)
                .build()?
                .completions_api()
                .agent(env::var("OPENAI_MODEL").unwrap_or_default())
                .preamble("You are a Discord bot named Blinky. Respond like a discord user.")
                .build(),
        );

        let gemini_agent = {
            let client = gemini::Client::builder()
                .api_key(env::var("GEMINI_API_KEY")?)
                .build()?;
            Arc::new(
                client
                    .agent(env::var("GEMINI_MODEL").unwrap_or_default().to_string())
                    .preamble("You are a Discord bot named Blinky. Respond like a discord user.")
                    .build(),
            )
        };

        let mut guard = state.lock().await;
        *guard = Some(State {
            bot_id: ready.user.id,
            openai_agent,
            gemini_agent,
            histories: Arc::new(RwLock::new(HashMap::new())),
        });

        tracing::info!("Shard is ready: {}", ready.user.name);
        return Ok(());
    }

    let (openai_agent, gemini_agent, histories, bot_id) = {
        let guard = state.lock().await;
        if let Ok(s) = guard.as_ref().context("State not initialized") {
            (
                s.openai_agent.clone(),
                s.gemini_agent.clone(),
                Arc::clone(&s.histories),
                s.bot_id,
            )
        } else {
            return Ok(());
        }
    };

    let model_provider = env::var("MODEL_PROVIDER").unwrap_or_default();

    if let Event::MessageCreate(msg) = event
        && (msg.mentions.iter().any(|mention| mention.id == bot_id)
            || msg.referenced_message.is_some())
        && msg.author.id != bot_id
    {
        if let Some(referenced_message) = msg.referenced_message.as_ref()
            && referenced_message.author.id == bot_id
        {
            return Ok(());
        }
        tracing::debug!("Received message: {}", msg.content);

        let user_id = msg.author.id.to_string();
        let user_prompt = msg
            .content
            .replace(&format!("<@!{}>", bot_id), "")
            .replace(&format!("<@{}>", bot_id), "")
            .trim()
            .to_string();

        let mut history = {
            let read_guard = histories.read().unwrap();
            read_guard.get(&user_id).cloned().unwrap_or_default()
        };
        let response = match model_provider.as_str() {
            "openai" => {
                let agent = openai_agent.clone();
                agent.chat(&user_prompt, history.clone()).await?
            }
            "gemini" => {
                let agent = gemini_agent.clone();
                agent.chat(&user_prompt, history.clone()).await?
            }
            _ => {
                bail!("Unsupported model provider: {}", model_provider);
            }
        };

        history.push(Message::user(&user_prompt));
        history.push(Message::assistant(&response));

        {
            let mut write_guard = histories.write().unwrap();
            write_guard.insert(user_id, history);
        }

        http.create_message(msg.channel_id)
            .reply(msg.id)
            .content(&response)
            .await?;
    }

    Ok(())
}
