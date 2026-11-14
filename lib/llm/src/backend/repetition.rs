// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend repetition stop condition (Axiom downstream).
//!
//! Stops a choice whose generated token IDs end in one N-gram repeated back to back,
//! using the same rule as vLLM's `SamplingParams.repetition_detection`: the last
//! `L * min_count` tokens are the last `L` tokens repeated `min_count` times, for some
//! pattern length `L` in `[min_pattern_size, max_pattern_size]`. Running it in the
//! frontend `Decoder` makes it engine-agnostic; the decoder's stop trigger already
//! stops the upstream engine request.
//!
//! Configured once per frontend process through `DYN_REPETITION_DETECTION`, either one
//! rule object or an array of rules (a "tier" per pattern-length range). Unset, empty,
//! or invalid values leave detection disabled.

use std::{collections::VecDeque, sync::Arc, sync::LazyLock};

use serde::Deserialize;

use crate::protocols::TokenIdType;
use dynamo_runtime::config::environment_names::llm as env_llm;

/// Largest pattern length accepted. Per-token work and memory are linear in the largest
/// configured pattern length, so it is bounded to keep the decoder hot path cheap.
pub(crate) const MAX_PATTERN_SIZE_LIMIT: usize = 4096;

/// One repetition rule, in vLLM's `RepetitionDetectionParams` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepetitionRule {
    /// Smallest pattern length checked. `0` means `1`.
    #[serde(default)]
    pub min_pattern_size: usize,
    /// Largest pattern length checked. Must be at least 1.
    pub max_pattern_size: usize,
    /// Consecutive occurrences needed to stop. Must be at least 2.
    pub min_count: usize,
}

impl RepetitionRule {
    fn min_len(&self) -> usize {
        self.min_pattern_size.max(1)
    }

    fn validate(&self) -> Result<(), String> {
        if self.max_pattern_size == 0 {
            return Err("max_pattern_size must be at least 1".to_string());
        }
        if self.max_pattern_size > MAX_PATTERN_SIZE_LIMIT {
            return Err(format!(
                "max_pattern_size must be at most {MAX_PATTERN_SIZE_LIMIT}, got {}",
                self.max_pattern_size
            ));
        }
        if self.min_len() > self.max_pattern_size {
            return Err(format!(
                "min_pattern_size ({}) must not exceed max_pattern_size ({})",
                self.min_pattern_size, self.max_pattern_size
            ));
        }
        if self.min_count < 2 {
            return Err(format!(
                "min_count must be at least 2, got {}",
                self.min_count
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawConfig {
    One(RepetitionRule),
    Many(Vec<RepetitionRule>),
}

/// Validated rules, compiled into one run-length threshold per pattern length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepetitionConfig {
    rules: Vec<RepetitionRule>,
    /// `thresholds[l]` is the run of `token[i] == token[i - l]` matches that stops a
    /// choice for pattern length `l`, or `usize::MAX` when no rule covers `l`.
    thresholds: Vec<usize>,
}

impl RepetitionConfig {
    pub fn new(rules: Vec<RepetitionRule>) -> Result<Self, String> {
        if rules.is_empty() {
            return Err("at least one rule is required".to_string());
        }
        for rule in &rules {
            rule.validate()?;
        }
        let max_len = rules
            .iter()
            .map(|rule| rule.max_pattern_size)
            .max()
            .unwrap_or(0);
        let mut thresholds = vec![usize::MAX; max_len + 1];
        for rule in &rules {
            for (len, threshold) in thresholds
                .iter_mut()
                .enumerate()
                .take(rule.max_pattern_size + 1)
                .skip(rule.min_len())
            {
                // `L` tokens repeated `c` times means the last `L * (c - 1)` positions each
                // equal the token `L` positions earlier.
                *threshold = (*threshold).min(len * (rule.min_count - 1));
            }
        }
        Ok(Self { rules, thresholds })
    }

    /// Parses one rule object or an array of rules.
    pub fn parse(value: &str) -> Result<Self, String> {
        let raw: RawConfig = serde_json::from_str(value).map_err(|error| error.to_string())?;
        Self::new(match raw {
            RawConfig::One(rule) => vec![rule],
            RawConfig::Many(rules) => rules,
        })
    }

    pub fn rules(&self) -> &[RepetitionRule] {
        &self.rules
    }

    fn max_pattern_size(&self) -> usize {
        self.thresholds.len() - 1
    }
}

static CONFIGURED: LazyLock<Option<Arc<RepetitionConfig>>> = LazyLock::new(|| {
    let value = std::env::var(env_llm::DYN_REPETITION_DETECTION).ok()?;
    if value.trim().is_empty() {
        return None;
    }
    match RepetitionConfig::parse(&value) {
        Ok(config) => {
            tracing::info!(
                rules = ?config.rules(),
                "frontend repetition detection enabled"
            );
            Some(Arc::new(config))
        }
        Err(error) => {
            tracing::error!(
                %error,
                value,
                "invalid {}; repetition detection is disabled",
                env_llm::DYN_REPETITION_DETECTION
            );
            None
        }
    }
});

/// The process-wide configuration from `DYN_REPETITION_DETECTION`, read once.
pub(crate) fn configured() -> Option<Arc<RepetitionConfig>> {
    CONFIGURED.clone()
}

/// Per-choice detector. `observe` is O(max_pattern_size) per token.
#[derive(Debug, Clone)]
pub(crate) struct RepetitionDetector {
    config: Arc<RepetitionConfig>,
    /// The last `max_pattern_size` generated tokens, oldest first.
    history: VecDeque<TokenIdType>,
    /// `runs[l]`: consecutive trailing positions whose token equals the token `l` earlier.
    runs: Vec<usize>,
}

impl RepetitionDetector {
    pub(crate) fn new(config: Arc<RepetitionConfig>) -> Self {
        let max_len = config.max_pattern_size();
        Self {
            history: VecDeque::with_capacity(max_len),
            runs: vec![0; max_len + 1],
            config,
        }
    }

    /// Records one generated token. Returns true when the output now ends in a
    /// configured repetition.
    pub(crate) fn observe(&mut self, token_id: TokenIdType) -> bool {
        let seen = self.history.len();
        let mut detected = false;
        for len in 1..self.runs.len() {
            let run = &mut self.runs[len];
            if len <= seen && self.history[seen - len] == token_id {
                *run += 1;
                if *run >= self.config.thresholds[len] {
                    detected = true;
                }
            } else {
                *run = 0;
            }
        }
        if self.history.len() == self.config.max_pattern_size() {
            self.history.pop_front();
        }
        self.history.push_back(token_id);
        detected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(min: usize, max: usize, count: usize) -> RepetitionRule {
        RepetitionRule {
            min_pattern_size: min,
            max_pattern_size: max,
            min_count: count,
        }
    }

    /// vLLM's `check_sequence_repetition`, transcribed, as the reference.
    fn reference(tokens: &[TokenIdType], rule: &RepetitionRule) -> bool {
        for len in rule.min_len()..=rule.max_pattern_size {
            if len * rule.min_count > tokens.len() {
                return false;
            }
            let repeated = (1..=len).all(|n| {
                let target = tokens[tokens.len() - n];
                (1..rule.min_count).all(|m| tokens[tokens.len() - (len * m + n)] == target)
            });
            if repeated {
                return true;
            }
        }
        false
    }

    /// First prefix length at which a detector over `rules` fires.
    fn first_hit(tokens: &[TokenIdType], rules: Vec<RepetitionRule>) -> Option<usize> {
        let mut detector = RepetitionDetector::new(Arc::new(RepetitionConfig::new(rules).unwrap()));
        tokens
            .iter()
            .position(|&token| detector.observe(token))
            .map(|index| index + 1)
    }

    #[test]
    fn detects_repeated_phrase_at_the_count() {
        let phrase = [7, 8, 9];
        let tokens: Vec<_> = phrase.iter().copied().cycle().take(9).collect();
        assert_eq!(first_hit(&tokens, vec![rule(3, 3, 3)]), Some(9));
        assert_eq!(first_hit(&tokens, vec![rule(3, 3, 4)]), None);
    }

    #[test]
    fn respects_pattern_length_bounds() {
        let single: Vec<_> = std::iter::repeat_n(5, 10).collect();
        // A single repeated token is also a repeated 2-gram; a range starting at 3 still
        // catches it as a 3-gram once nine tokens are out.
        assert_eq!(first_hit(&single, vec![rule(3, 4, 3)]), Some(9));
        assert_eq!(first_hit(&single, vec![rule(1, 1, 3)]), Some(3));
    }

    #[test]
    fn non_repeating_output_never_fires() {
        let tokens: Vec<_> = (0..500).collect();
        assert_eq!(first_hit(&tokens, vec![rule(1, 64, 2)]), None);
    }

    #[test]
    fn tiers_use_the_smallest_threshold_per_length() {
        let short = rule(1, 4, 16);
        let long = rule(5, 32, 3);
        let loop5: Vec<_> = [1, 2, 3, 4, 6].iter().copied().cycle().take(15).collect();
        assert_eq!(first_hit(&loop5, vec![short, long]), Some(15));
        let loop2: Vec<_> = [1, 2].iter().copied().cycle().take(40).collect();
        // A 2-token loop is first caught by the long tier as a 6-token pattern x3.
        assert_eq!(first_hit(&loop2, vec![short, long]), Some(18));
    }

    #[test]
    fn matches_the_vllm_reference_on_random_sequences() {
        // Small alphabets make accidental repeats common, which exercises run resets.
        let mut state = 0x9e37_79b9_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for case in 0..400 {
            let rule = rule(
                (next() % 4) as usize,
                1 + (next() % 12) as usize,
                2 + (next() % 4) as usize,
            );
            let rule = RepetitionRule {
                min_pattern_size: rule.min_pattern_size.min(rule.max_pattern_size),
                ..rule
            };
            let alphabet = 2 + (next() % 3);
            let tokens: Vec<TokenIdType> = (0..120)
                .map(|_| (next() % alphabet) as TokenIdType)
                .collect();
            let mut detector =
                RepetitionDetector::new(Arc::new(RepetitionConfig::new(vec![rule]).unwrap()));
            for end in 1..=tokens.len() {
                assert_eq!(
                    detector.observe(tokens[end - 1]),
                    reference(&tokens[..end], &rule),
                    "case {case}, rule {rule:?}, prefix {end}: {:?}",
                    &tokens[..end]
                );
            }
        }
    }

    #[test]
    fn parses_one_rule_or_a_list() {
        let one = RepetitionConfig::parse(
            r#"{"min_pattern_size": 8, "max_pattern_size": 256, "min_count": 4}"#,
        )
        .unwrap();
        assert_eq!(one.rules(), &[rule(8, 256, 4)]);
        let many = RepetitionConfig::parse(
            r#"[{"max_pattern_size": 7, "min_count": 32},
                {"min_pattern_size": 8, "max_pattern_size": 256, "min_count": 4}]"#,
        )
        .unwrap();
        assert_eq!(many.rules(), &[rule(0, 7, 32), rule(8, 256, 4)]);
    }

    #[test]
    fn rejects_invalid_rules() {
        for value in [
            r#"{"max_pattern_size": 0, "min_count": 3}"#,
            r#"{"max_pattern_size": 8, "min_count": 1}"#,
            r#"{"min_pattern_size": 9, "max_pattern_size": 8, "min_count": 3}"#,
            r#"{"max_pattern_size": 4097, "min_count": 3}"#,
            r#"{"max_pattern_size": 8, "min_count": 3, "typo": 1}"#,
            r#"[]"#,
            r#"not json"#,
        ] {
            assert!(RepetitionConfig::parse(value).is_err(), "{value}");
        }
    }
}
