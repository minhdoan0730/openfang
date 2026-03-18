//! Reaction coordination for Tier 2‑3 of the 4‑tier processing ladder.
//!
//! Manages a time window for collecting reactions, decides a winner, and
//! optionally cleans up loser reactions.

use openfang_types::config::ReactionConfig;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;
use tracing::debug;

/// Winner decision after the coordination window closes.
#[derive(Debug, Clone)]
pub enum WinnerDecision {
    /// A single agent reacted → winner.
    SingleWinner(String),
    /// Multiple agents reacted → need domain classification (Tier 3).
    NeedsClassification(Vec<String>),
    /// No reactions within the window.
    NoWinner,
}

/// A single reaction event.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ReactionEvent {
    agent_id: String,
    emoji: String,
    timestamp: Instant,
}

/// Manages the coordination window for a single message.
pub struct ReactionCoordinator {
    config: ReactionConfig,
    /// Active windows keyed by message ID.
    windows: RwLock<HashMap<String, Arc<RwLock<CoordinationWindow>>>>,
}

#[allow(dead_code)]
struct CoordinationWindow {
    message_id: String,
    reactions: Vec<ReactionEvent>,
    created_at: Instant,
    expired: bool,
}

impl ReactionCoordinator {
    /// Create a new coordinator with the given configuration.
    pub fn new(config: ReactionConfig) -> Self {
        Self {
            config,
            windows: RwLock::new(HashMap::new()),
        }
    }

    /// Record a reaction and start the coordination window if not already open.
    pub async fn record_reaction(
        &self,
        message_id: String,
        agent_id: String,
        emoji: String,
    ) -> bool {
        let window = self.get_or_create_window(&message_id).await;
        let mut window = window.write().await;
        if window.expired {
            return false;
        }
        window.reactions.push(ReactionEvent {
            agent_id,
            emoji,
            timestamp: Instant::now(),
        });
        true
    }

    /// Get the winner decision after the window closes.
    /// This should be called after waiting for the coordination window.
    pub async fn get_winner(&self, message_id: &str) -> WinnerDecision {
        let window = match self.windows.read().await.get(message_id) {
            Some(w) => w.clone(),
            None => return WinnerDecision::NoWinner,
        };
        let window = window.read().await;
        if window.expired || window.reactions.is_empty() {
            return WinnerDecision::NoWinner;
        }

        // Collect unique agents
        let agents: HashSet<String> = window
            .reactions
            .iter()
            .map(|r| r.agent_id.clone())
            .collect();

        match agents.len() {
            0 => WinnerDecision::NoWinner,
            1 => {
                let winner = agents.into_iter().next().unwrap();
                WinnerDecision::SingleWinner(winner)
            }
            _ => WinnerDecision::NeedsClassification(agents.into_iter().collect()),
        }
    }

    /// Expire a window manually (e.g., after a decision has been made).
    pub async fn expire_window(&self, message_id: &str) {
        if let Some(window) = self.windows.write().await.get_mut(message_id) {
            window.write().await.expired = true;
        }
        self.cleanup_windows().await;
    }

    /// Start a background task that expires windows after the coordination window.
    pub async fn start_background_expiry(&self) {
        let coordinator = self.clone();
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_millis(coordinator.config.coordination_window_ms)).await;
                coordinator.cleanup_windows().await;
            }
        });
    }

    // --- internal helpers ---

    async fn get_or_create_window(&self, message_id: &str) -> Arc<RwLock<CoordinationWindow>> {
        let mut windows = self.windows.write().await;
        if let Some(window) = windows.get(message_id) {
            return window.clone();
        }
        let window = Arc::new(RwLock::new(CoordinationWindow {
            message_id: message_id.to_string(),
            reactions: Vec::new(),
            created_at: Instant::now(),
            expired: false,
        }));
        windows.insert(message_id.to_string(), window.clone());
        debug!(message_id, "Started reaction coordination window");
        window
    }

    async fn cleanup_windows(&self) {
        let mut windows = self.windows.write().await;
        let expired = windows
            .iter()
            .filter(|(_, w)| {
                match w.try_read() {
                    Ok(w) => w.expired || w.created_at.elapsed() >= Duration::from_millis(self.config.coordination_window_ms),
                    Err(_) => true, // remove poisoned entries
                }
            })
            .map(|(k, _)| k.clone())
            .collect::<Vec<_>>();
        for id in &expired {
            windows.remove(id);
        }
        if !expired.is_empty() {
            debug!(expired = expired.len(), "Cleaned up expired reaction windows");
        }
    }
}

impl Clone for ReactionCoordinator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            windows: RwLock::new(HashMap::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ReactionConfig {
        ReactionConfig {
            coordination_window_ms: 400,
            classifier_model: "claude-3-haiku-20240307".to_string(),
            classifier_threshold: 0.70,
            cleanup_reactions: true,
        }
    }

    #[tokio::test]
    async fn test_single_winner() {
        let coordinator = ReactionCoordinator::new(test_config());
        coordinator
            .record_reaction("msg1".to_string(), "agent1".to_string(), "👍".to_string())
            .await;
        // Wait a bit for window to still be open
        sleep(Duration::from_millis(10)).await;
        match coordinator.get_winner("msg1").await {
            WinnerDecision::SingleWinner(id) => assert_eq!(id, "agent1"),
            _ => panic!("Expected SingleWinner"),
        }
    }

    #[tokio::test]
    async fn test_no_reactions() {
        let coordinator = ReactionCoordinator::new(test_config());
        match coordinator.get_winner("msg2").await {
            WinnerDecision::NoWinner => (),
            _ => panic!("Expected NoWinner"),
        }
    }

    #[tokio::test]
    async fn test_needs_classification() {
        let coordinator = ReactionCoordinator::new(test_config());
        coordinator
            .record_reaction("msg3".to_string(), "agent1".to_string(), "👍".to_string())
            .await;
        coordinator
            .record_reaction("msg3".to_string(), "agent2".to_string(), "👍".to_string())
            .await;
        sleep(Duration::from_millis(10)).await;
        match coordinator.get_winner("msg3").await {
            WinnerDecision::NeedsClassification(agents) => {
                assert_eq!(agents.len(), 2);
                assert!(agents.contains(&"agent1".to_string()));
                assert!(agents.contains(&"agent2".to_string()));
            }
            _ => panic!("Expected NeedsClassification"),
        }
    }
}