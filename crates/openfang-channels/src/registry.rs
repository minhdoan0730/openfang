//! Adapter registry for multi‑agent Discord setups.
//!
//! Maps (channel type, agent ID) to `ChannelAdapter` instances, allowing
//! `RelayReply` mode to find the correct bot token for posting.

use crate::types::{ChannelAdapter, ChannelType};
use dashmap::DashMap;
use openfang_memory::MemorySubstrate;
use openfang_types::agent::AgentId;
use openfang_types::config::{BoardroomConfig, TurnPolicy};
use openfang_types::error::{OpenFangError, OpenFangResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::Uuid;

/// Registry of channel adapters keyed by channel type and agent ID.
pub struct AdapterRegistry {
    adapters: DashMap<(ChannelType, String), Arc<dyn ChannelAdapter>>,
}

impl AdapterRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            adapters: DashMap::new(),
        }
    }

    /// Register an adapter for a specific agent.
    pub fn register(
        &self,
        channel_type: ChannelType,
        agent_id: String,
        adapter: Arc<dyn ChannelAdapter>,
    ) {
        self.adapters.insert((channel_type, agent_id), adapter);
    }

    /// Retrieve an adapter for a given channel type and agent ID.
    pub fn get(
        &self,
        channel_type: ChannelType,
        agent_id: &str,
    ) -> Option<Arc<dyn ChannelAdapter>> {
        self.adapters
            .get(&(channel_type, agent_id.to_string()))
            .map(|entry| entry.value().clone())
    }

    /// Remove an adapter from the registry.
    pub fn remove(&self, channel_type: ChannelType, agent_id: &str) -> bool {
        self.adapters
            .remove(&(channel_type, agent_id.to_string()))
            .is_some()
    }

    /// Get all adapters for a given channel type.
    pub fn get_by_channel(&self, channel_type: ChannelType) -> Vec<Arc<dyn ChannelAdapter>> {
        self.adapters
            .iter()
            .filter(|entry| entry.key().0 == channel_type)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get the number of registered adapters.
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}
/// Agent ID used for storing boardroom thread state in the memory substrate.
const BOARDROOM_AGENT_ID: AgentId = AgentId(Uuid::from_u128(0));

/// Lifecycle state of a boardroom thread.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThreadStatus {
    /// Thread created but no agents have joined yet.
    Open,
    /// At least one agent has joined and the thread is active.
    Active,
    /// Thread has been inactive for longer than `idle_timeout_minutes`.
    Idle,
    /// Thread has been explicitly closed or archived.
    Closed,
}

/// Persistent state of a single boardroom thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardroomThread {
    /// Discord thread ID (channel ID).
    pub thread_id: String,
    /// IDs of agents currently participating.
    pub participants: HashSet<String>,
    /// Current lifecycle status.
    pub status: ThreadStatus,
    /// Turn policy for this thread.
    pub turn_policy: TurnPolicy,
    /// Number of consecutive bot messages (circuit breaker).
    pub bot_streak: u32,
    /// Timestamp of last activity (message or reaction).
    pub last_activity: DateTime<Utc>,
}

impl BoardroomThread {
    /// Create a new thread with default state.
    pub fn new(thread_id: String, turn_policy: TurnPolicy) -> Self {
        Self {
            thread_id,
            participants: HashSet::new(),
            status: ThreadStatus::Open,
            turn_policy,
            bot_streak: 0,
            last_activity: Utc::now(),
        }
    }

    /// Record a bot message, incrementing the streak.
    pub fn record_bot_message(&mut self) {
        self.bot_streak += 1;
        self.last_activity = Utc::now();
    }

    /// Record a human message, resetting the streak.
    pub fn record_human_message(&mut self) {
        self.bot_streak = 0;
        self.last_activity = Utc::now();
    }

    /// Add a participant agent.
    pub fn add_participant(&mut self, agent_id: String) {
        self.participants.insert(agent_id);
        if self.status == ThreadStatus::Open && !self.participants.is_empty() {
            self.status = ThreadStatus::Active;
        }
    }

    /// Remove a participant.
    pub fn remove_participant(&mut self, agent_id: &str) {
        self.participants.remove(agent_id);
        if self.participants.is_empty() {
            self.status = ThreadStatus::Idle;
        }
    }

    /// Check whether the circuit breaker has tripped.
    pub fn circuit_breaker_tripped(&self, max_bot_streak: u32) -> bool {
        self.bot_streak >= max_bot_streak
    }

    /// Check whether the thread is idle (inactive longer than `timeout`).
    pub fn is_idle(&self, idle_timeout: Duration) -> bool {
        let elapsed = Utc::now() - self.last_activity;
        elapsed >= chrono::Duration::from_std(idle_timeout).unwrap_or_default()
    }
}

/// Registry of all boardroom threads, with persistence via `openfang‑memory`.
pub struct BoardroomRegistry {
    config: BoardroomConfig,
    threads: RwLock<HashMap<String, Arc<RwLock<BoardroomThread>>>>,
    memory: Arc<MemorySubstrate>,
}

impl BoardroomRegistry {
    /// Create a new registry with the given configuration and memory substrate.
    /// Loads existing thread state from memory.
    pub async fn new(config: BoardroomConfig, memory: Arc<MemorySubstrate>) -> OpenFangResult<Self> {
        let mut registry = Self {
            config,
            threads: RwLock::new(HashMap::new()),
            memory,
        };
        registry.load_from_memory().await?;
        Ok(registry)
    }

    /// Load all persisted threads from the memory substrate.
    async fn load_from_memory(&mut self) -> OpenFangResult<()> {
        // List all KV pairs for the boardroom agent
        let kvs = self.memory.list_kv(BOARDROOM_AGENT_ID)?;

        let mut threads = HashMap::new();
        for (key, value) in kvs {
            if let Some(thread_id) = key.strip_prefix("boardroom:thread:") {
                let thread: BoardroomThread = serde_json::from_value(value)
                    .map_err(|e| OpenFangError::Memory(format!("Failed to deserialize thread {}: {}", thread_id, e)))?;
                threads.insert(thread_id.to_string(), Arc::new(RwLock::new(thread)));
            }
        }

        let loaded_count = threads.len();
        *self.threads.write().await = threads;
        info!(loaded_count, "Loaded boardroom threads from memory");
        Ok(())
    }

    /// Save a single thread to the memory substrate.
    async fn save_thread(&self, thread_id: &str, thread: &BoardroomThread) -> OpenFangResult<()> {
        let key = format!("boardroom:thread:{}", thread_id);
        let value = serde_json::to_value(thread)
            .map_err(|e| OpenFangError::Memory(format!("Failed to serialize thread {}: {}", thread_id, e)))?;
        self.memory.structured_set(BOARDROOM_AGENT_ID, &key, value)?;
        Ok(())
    }

    /// Remove a thread from the memory substrate.
    async fn delete_thread(&self, thread_id: &str) -> OpenFangResult<()> {
        let key = format!("boardroom:thread:{}", thread_id);
        self.memory.structured_delete(BOARDROOM_AGENT_ID, &key)?;
        Ok(())
    }

    /// Get or create a thread.
    pub async fn get_or_create_thread(
        &self,
        thread_id: String,
        turn_policy: Option<TurnPolicy>,
    ) -> OpenFangResult<Arc<RwLock<BoardroomThread>>> {
        let policy = turn_policy.unwrap_or(self.config.default_turn_policy);
        let mut threads = self.threads.write().await;
        if let Some(thread) = threads.get(&thread_id) {
            return Ok(thread.clone());
        }
        let thread = Arc::new(RwLock::new(BoardroomThread::new(thread_id.clone(), policy)));
        // Save to memory
        self.save_thread(&thread_id, &*thread.read().await).await?;
        threads.insert(thread_id.clone(), thread.clone());
        info!(thread_id, "Created new boardroom thread");
        Ok(thread)
    }

    /// Get a thread if it exists.
    pub async fn get_thread(&self, thread_id: &str) -> Option<Arc<RwLock<BoardroomThread>>> {
        let threads = self.threads.read().await;
        threads.get(thread_id).cloned()
    }

    /// Remove a thread (e.g., after closure or idle cleanup).
    pub async fn remove_thread(&self, thread_id: &str) -> OpenFangResult<bool> {
        let mut threads = self.threads.write().await;
        let existed = threads.remove(thread_id).is_some();
        if existed {
            self.delete_thread(thread_id).await?;
        }
        Ok(existed)
    }

    /// Add a participant to a thread.
    pub async fn add_participant(&self, thread_id: &str, agent_id: String) -> OpenFangResult<bool> {
        if let Some(thread) = self.get_thread(thread_id).await {
            let mut thread = thread.write().await;
            thread.add_participant(agent_id);
            self.save_thread(thread_id, &thread).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Record a bot message in a thread, tripping the circuit breaker if needed.
    /// Returns `true` if the circuit breaker has tripped (all agents should be silenced).
    pub async fn record_bot_message(&self, thread_id: &str) -> OpenFangResult<bool> {
        let thread = match self.get_thread(thread_id).await {
            Some(t) => t,
            None => return Ok(false),
        };
        let mut thread = thread.write().await;
        thread.record_bot_message();
        let tripped = thread.circuit_breaker_tripped(self.config.max_bot_streak);
        if tripped {
            warn!(
                thread_id,
                max_bot_streak = self.config.max_bot_streak,
                "Boardroom circuit breaker tripped — silencing all agents"
            );
        }
        self.save_thread(thread_id, &thread).await?;
        Ok(tripped)
    }

    /// Record a human message, resetting the circuit breaker.
    pub async fn record_human_message(&self, thread_id: &str) -> OpenFangResult<()> {
        if let Some(thread) = self.get_thread(thread_id).await {
            let mut thread = thread.write().await;
            thread.record_human_message();
            self.save_thread(thread_id, &thread).await?;
        }
        Ok(())
    }

    /// Run periodic cleanup of idle threads.
    pub async fn cleanup_idle_threads(&self) -> OpenFangResult<usize> {
        let idle_timeout = Duration::from_secs(self.config.idle_timeout_minutes as u64 * 60);
        let mut threads = self.threads.write().await;
        let before = threads.len();
        let mut to_delete = Vec::new();
        threads.retain(|thread_id, thread| {
            match thread.try_read() {
                Ok(thread) => {
                    if thread.is_idle(idle_timeout) {
                        to_delete.push(thread_id.clone());
                        false
                    } else {
                        true
                    }
                }
                Err(_) => {
                    // remove poisoned entry
                    to_delete.push(thread_id.clone());
                    false
                }
            }
        });
        // Delete from memory
        for thread_id in &to_delete {
            let _ = self.delete_thread(thread_id).await;
        }
        let removed = before - threads.len();
        if removed > 0 {
            info!(removed, "Cleaned up idle boardroom threads");
        }
        Ok(removed)
    }

    /// Get the number of active threads.
    pub async fn thread_count(&self) -> usize {
        let threads = self.threads.read().await;
        threads.len()
    }
}
