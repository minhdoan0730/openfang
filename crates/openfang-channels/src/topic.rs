//! Topic filter for Tier 1 of the 4‑tier processing ladder.
//!
//! Filters messages based on keyword matching and length thresholds.
//! Used to decide whether a message is relevant to an agent’s domain.

use openfang_types::config::TopicFilterConfig;
use std::fmt;

/// Relevance decision produced by the topic filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Relevance {
    /// Message is relevant (contains positive keywords, no negative overrides).
    Yes,
    /// Message is not relevant (contains negative keywords, or too short).
    No,
    /// Unable to decide (no keywords match, length above threshold).
    Unknown,
}

impl fmt::Display for Relevance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Relevance::Yes => write!(f, "Yes"),
            Relevance::No => write!(f, "No"),
            Relevance::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Topic filter for a single agent.
#[derive(Clone)]
pub struct TopicFilter {
    config: TopicFilterConfig,
}

impl TopicFilter {
    /// Create a new filter from configuration.
    pub fn new(config: TopicFilterConfig) -> Self {
        Self { config }
    }

    /// Evaluate a message’s relevance to this agent’s domain.
    pub fn evaluate(&self, message: &str) -> Relevance {
        // Very short messages are considered unknown (Tier 1 passes them through).
        if message.len() < self.config.min_length {
            return Relevance::Unknown;
        }

        let lower = message.to_lowercase();

        // Negative keywords override everything.
        for kw in &self.config.negative_keywords {
            if lower.contains(&kw.to_lowercase()) {
                return Relevance::No;
            }
        }

        // Positive keywords indicate relevance.
        let mut has_positive = false;
        for kw in &self.config.keywords {
            if lower.contains(&kw.to_lowercase()) {
                has_positive = true;
                break;
            }
        }

        if has_positive {
            Relevance::Yes
        } else {
            Relevance::Unknown
        }
    }

    /// Get the minimum length threshold.
    pub fn min_length(&self) -> usize {
        self.config.min_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TopicFilterConfig {
        TopicFilterConfig {
            keywords: vec!["rust".to_string(), "programming".to_string()],
            negative_keywords: vec!["not rust".to_string(), "ignore".to_string()],
            min_length: 10,
        }
    }

    #[test]
    fn test_positive_match() {
        let filter = TopicFilter::new(test_config());
        assert_eq!(
            filter.evaluate("I love Rust programming"),
            Relevance::Yes
        );
        assert_eq!(
            filter.evaluate("Programming is fun"),
            Relevance::Yes
        );
    }

    #[test]
    fn test_negative_override() {
        let filter = TopicFilter::new(test_config());
        assert_eq!(
            filter.evaluate("This is not rust related"),
            Relevance::No
        );
        assert_eq!(
            filter.evaluate("Ignore this message"),
            Relevance::No
        );
    }

    #[test]
    fn test_unknown() {
        let filter = TopicFilter::new(test_config());
        assert_eq!(
            filter.evaluate("Hello world"), // no keywords
            Relevance::Unknown
        );
    }

    #[test]
    fn test_too_short() {
        let filter = TopicFilter::new(test_config());
        // Length 9 < min_length 10 → Unknown
        assert_eq!(filter.evaluate("Short"), Relevance::Unknown);
    }
}