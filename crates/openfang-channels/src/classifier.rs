//! Domain classifier for Tier 3 of the 4‑tier processing ladder.
//!
//! Resolves ambiguous reaction scenarios (2+ agents reacted) by calling a
//! lightweight LLM to pick the most appropriate agent based on message content.

use openfang_types::config::ReactionConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

/// Classification result from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassificationResponse {
    /// Selected agent ID.
    winner: String,
    /// Confidence score (0.0–1.0).
    confidence: f32,
    /// Reasoning (for debugging).
    reasoning: String,
}

/// Domain classifier for ambiguous reaction decisions.
pub struct DomainClassifier {
    config: ReactionConfig,
    /// Mock for testing; in production would be an LLM client.
    #[allow(dead_code)]
    mock_responses: HashMap<String, String>,
}

impl DomainClassifier {
    /// Create a new classifier with the given configuration.
    pub fn new(config: ReactionConfig) -> Self {
        Self {
            config,
            mock_responses: HashMap::new(),
        }
    }

    /// Classify the most appropriate agent for a message among candidates.
    /// Returns `Some(agent_id)` if confidence exceeds threshold, otherwise `None`.
    pub async fn classify(
        &self,
        message: &str,
        candidate_agents: &[String],
    ) -> Option<String> {
        if candidate_agents.is_empty() {
            return None;
        }
        if candidate_agents.len() == 1 {
            // Should not happen — caller should handle single‑winner case.
            return Some(candidate_agents[0].clone());
        }

        // TODO: Replace with actual LLM call.
        let response = self.mock_llm_call(message, candidate_agents).await;

        if response.confidence >= self.config.classifier_threshold {
            debug!(
                winner = response.winner,
                confidence = response.confidence,
                reasoning = response.reasoning,
                "Domain classifier selected winner"
            );
            Some(response.winner)
        } else {
            warn!(
                confidence = response.confidence,
                threshold = self.config.classifier_threshold,
                "Classifier confidence below threshold"
            );
            None
        }
    }

    /// Mock LLM call that returns a deterministic winner for testing.
    async fn mock_llm_call(&self, message: &str, candidates: &[String]) -> ClassificationResponse {
        // Simple deterministic rule: pick the first candidate whose ID appears in the message,
        // otherwise the first candidate.
        let lower = message.to_lowercase();
        for agent in candidates {
            if lower.contains(&agent.to_lowercase()) {
                return ClassificationResponse {
                    winner: agent.clone(),
                    confidence: 0.85,
                    reasoning: format!("Agent ID found in message text."),
                };
            }
        }

        ClassificationResponse {
            winner: candidates[0].clone(),
            confidence: 0.60,
            reasoning: "Fallback to first candidate.".to_string(),
        }
    }

    /// Set a mock response for a specific message (testing only).
    #[cfg(test)]
    pub fn set_mock_response(&mut self, message: String, winner: String) {
        self.mock_responses.insert(message, winner);
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
    async fn test_classify_above_threshold() {
        let classifier = DomainClassifier::new(test_config());
        let candidates = vec!["agent1".to_string(), "agent2".to_string()];
        let winner = classifier
            .classify("This is about agent1 stuff", &candidates)
            .await;
        assert_eq!(winner, Some("agent1".to_string()));
    }

    #[tokio::test]
    async fn test_classify_below_threshold() {
        let mut config = test_config();
        config.classifier_threshold = 0.90; // higher than mock confidence
        let classifier = DomainClassifier::new(config);
        let candidates = vec!["agent1".to_string(), "agent2".to_string()];
        let winner = classifier
            .classify("Unrelated message", &candidates)
            .await;
        assert_eq!(winner, None);
    }

    #[tokio::test]
    async fn test_single_candidate() {
        let classifier = DomainClassifier::new(test_config());
        let candidates = vec!["agent1".to_string()];
        let winner = classifier.classify("any message", &candidates).await;
        assert_eq!(winner, Some("agent1".to_string()));
    }
}