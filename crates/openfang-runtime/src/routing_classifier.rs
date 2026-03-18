//! Routing classifier for Discord multi‑agent boardroom.
//!
//! Analyzes messages and decides whether the default agent should handle,
//! delegate to a specialist, or start a boardroom thread.

use openfang_types::config::{RoutingRosterEntry, TurnPolicy, ReplyMode};
use serde_json::Value;
use tracing::{debug, warn};

/// Routing decision produced by the classifier.
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Handle directly (default agent replies).
    Handle,
    /// Delegate to a single specialist agent.
    Delegate(DelegateTarget),
    /// Delegate to multiple specialists (parallel execution).
    DelegateMulti(Vec<DelegateTarget>),
    /// Create a boardroom thread with one or more specialists.
    Boardroom(Vec<DelegateTarget>),
}

/// Target of a delegation.
#[derive(Debug, Clone)]
pub struct DelegateTarget {
    /// Agent ID of the specialist.
    pub agent_id: String,
    /// Task description to send to the specialist.
    pub task: String,
}

impl DelegateTarget {
    pub fn new(agent_id: String, task: String) -> Self {
        Self { agent_id, task }
    }
}

/// Classifies messages based on the agent’s routing roster and LLM signals.
pub struct RoutingClassifier {
    /// Routing roster entries (specialists this agent can delegate to).
    roster: Vec<RoutingRosterEntry>,
    /// Turn policy for boardroom threads created by this agent.
    turn_policy: TurnPolicy,
    /// Reply mode for delegated tasks.
    reply_mode: ReplyMode,
}

impl RoutingClassifier {
    /// Create a new classifier with the given configuration.
    pub fn new(
        roster: Vec<RoutingRosterEntry>,
        turn_policy: TurnPolicy,
        reply_mode: ReplyMode,
    ) -> Self {
        Self {
            roster,
            turn_policy,
            reply_mode,
        }
    }

    /// Classify a message, optionally using an LLM‑generated `ROUTE:` signal.
    ///
    /// If `llm_output` contains a `ROUTE:` line, it is parsed as JSON and used.
    /// Otherwise, falls back to keyword matching against the routing roster.
    pub fn classify(&self, message: &str, llm_output: Option<&str>) -> RoutingDecision {
        // First, try to parse a ROUTE: signal from the LLM output.
        if let Some(output) = llm_output {
            if let Some(route_signal) = extract_route_signal(output) {
                match parse_route_signal(&route_signal) {
                    Ok(decision) => {
                        debug!("Routing decision from LLM signal: {:?}", decision);
                        return decision;
                    }
                    Err(e) => {
                        warn!("Failed to parse ROUTE: signal: {}", e);
                        // fall through to keyword matching
                    }
                }
            }
        }

        // Fallback: keyword matching against routing roster.
        let lower = message.to_lowercase();
        let mut matches = Vec::new();
        for entry in &self.roster {
            for domain in &entry.domains {
                if lower.contains(&domain.to_lowercase()) {
                    matches.push(DelegateTarget::new(
                        entry.agent_id.clone(),
                        format!("Please help with: {}", message),
                    ));
                    break; // each agent at most once
                }
            }
        }

        match matches.len() {
            0 => RoutingDecision::Handle,
            1 => RoutingDecision::Delegate(matches.remove(0)),
            _ => RoutingDecision::DelegateMulti(matches),
        }
    }

    /// Get the turn policy for boardroom threads created by this agent.
    pub fn turn_policy(&self) -> TurnPolicy {
        self.turn_policy
    }

    /// Get the reply mode for delegated tasks.
    pub fn reply_mode(&self) -> ReplyMode {
        self.reply_mode
    }
}

/// Extract the first `ROUTE:` line from LLM output.
fn extract_route_signal(llm_output: &str) -> Option<String> {
    for line in llm_output.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("ROUTE:") {
            return Some(trimmed[6..].trim().to_string());
        }
    }
    None
}

/// Parse a ROUTE: signal JSON into a `RoutingDecision`.
fn parse_route_signal(signal: &str) -> Result<RoutingDecision, String> {
    let value: Value = serde_json::from_str(signal)
        .map_err(|e| format!("Invalid JSON in ROUTE: signal: {}", e))?;

    let decision_type = value["decision"].as_str()
        .ok_or_else(|| "Missing 'decision' field".to_string())?;

    match decision_type {
        "handle" => Ok(RoutingDecision::Handle),
        "delegate" => {
            let agent_id = value["agent_id"].as_str()
                .ok_or_else(|| "Missing 'agent_id' for delegate".to_string())?;
            let task = value["task"].as_str()
                .unwrap_or("Please help with this.");
            Ok(RoutingDecision::Delegate(DelegateTarget::new(
                agent_id.to_string(),
                task.to_string(),
            )))
        }
        "delegate_multi" => {
            let agents = value["agents"].as_array()
                .ok_or_else(|| "Missing 'agents' array for delegate_multi".to_string())?;
            let mut targets = Vec::new();
            for agent in agents {
                let agent_id = agent["agent_id"].as_str()
                    .ok_or_else(|| "Missing agent_id in delegate_multi entry".to_string())?;
                let task = agent["task"].as_str()
                    .unwrap_or("Please help with this.");
                targets.push(DelegateTarget::new(agent_id.to_string(), task.to_string()));
            }
            Ok(RoutingDecision::DelegateMulti(targets))
        }
        "boardroom" => {
            let agents = value["agents"].as_array()
                .ok_or_else(|| "Missing 'agents' array for boardroom".to_string())?;
            let mut targets = Vec::new();
            for agent in agents {
                let agent_id = agent["agent_id"].as_str()
                    .ok_or_else(|| "Missing agent_id in boardroom entry".to_string())?;
                let task = agent["task"].as_str()
                    .unwrap_or("Please help with this.");
                targets.push(DelegateTarget::new(agent_id.to_string(), task.to_string()));
            }
            Ok(RoutingDecision::Boardroom(targets))
        }
        _ => Err(format!("Unknown decision type: {}", decision_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::config::{RoutingRosterEntry, TurnPolicy, ReplyMode};

    fn test_roster() -> Vec<RoutingRosterEntry> {
        vec![
            RoutingRosterEntry {
                agent_id: "frontend".to_string(),
                role: "Frontend developer".to_string(),
                domains: vec!["javascript".to_string(), "react".to_string()],
            },
            RoutingRosterEntry {
                agent_id: "backend".to_string(),
                role: "Backend developer".to_string(),
                domains: vec!["rust".to_string(), "api".to_string()],
            },
        ]
    }

    #[test]
    fn test_extract_route_signal() {
        let output = "Some text\nROUTE: {\"decision\":\"delegate\"}\nMore text";
        let signal = extract_route_signal(output).unwrap();
        assert_eq!(signal, "{\"decision\":\"delegate\"}");
    }

    #[test]
    fn test_parse_route_signal_handle() {
        let signal = r#"{"decision":"handle"}"#;
        let decision = parse_route_signal(signal).unwrap();
        assert!(matches!(decision, RoutingDecision::Handle));
    }

    #[test]
    fn test_parse_route_signal_delegate() {
        let signal = r#"{"decision":"delegate","agent_id":"frontend","task":"Fix the bug"}"#;
        let decision = parse_route_signal(signal).unwrap();
        match decision {
            RoutingDecision::Delegate(target) => {
                assert_eq!(target.agent_id, "frontend");
                assert_eq!(target.task, "Fix the bug");
            }
            _ => panic!("Expected Delegate"),
        }
    }

    #[test]
    fn test_classify_with_llm_signal() {
        let classifier = RoutingClassifier::new(
            test_roster(),
            TurnPolicy::MentionOnly,
            ReplyMode::DirectReply,
        );
        let llm_output = "ROUTE: {\"decision\":\"delegate\",\"agent_id\":\"frontend\",\"task\":\"Fix UI\"}";
        let decision = classifier.classify("ignore", Some(llm_output));
        match decision {
            RoutingDecision::Delegate(target) => {
                assert_eq!(target.agent_id, "frontend");
            }
            _ => panic!("Expected Delegate from LLM signal"),
        }
    }

    #[test]
    fn test_classify_keyword_match() {
        let classifier = RoutingClassifier::new(
            test_roster(),
            TurnPolicy::MentionOnly,
            ReplyMode::DirectReply,
        );
        let decision = classifier.classify("I need help with Rust programming", None);
        match decision {
            RoutingDecision::Delegate(target) => {
                assert_eq!(target.agent_id, "backend");
            }
            _ => panic!("Expected Delegate from keyword match"),
        }
    }
}