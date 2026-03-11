//! Adapter registry for multi‑agent Discord setups.
//!
//! Maps (channel type, agent ID) to `ChannelAdapter` instances, allowing
//! `RelayReply` mode to find the correct bot token for posting.

use crate::types::{ChannelAdapter, ChannelType};
use dashmap::DashMap;
use std::sync::Arc;

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