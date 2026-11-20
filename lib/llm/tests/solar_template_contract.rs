// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

fn render(messages: Value, effort: &str) -> Result<String, minijinja::Error> {
    let mut env = minijinja::Environment::new();
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    env.add_template(
        "solar",
        include_str!("../../../deploy/templates/solar_open2.jinja"),
    )?;
    env.get_template("solar")?.render(json!({
        "messages": messages, "reasoning_effort": effort, "add_generation_prompt": true
    }))
}

#[test]
fn solar_generation_prefix_matches_effort() {
    for effort in ["medium", "high", "xhigh"] {
        let text = render(json!([{"role":"user","content":"hello"}]), effort).unwrap();
        assert!(text.ends_with("<|think:start|>"), "{effort}");
    }
    for effort in ["none", "low"] {
        let text = render(json!([{"role":"user","content":"hello"}]), effort).unwrap();
        assert!(text.ends_with("<|think:start|><|think:end|>"), "{effort}");
    }
}

#[test]
fn solar_rejects_missing_tool_ids() {
    for id in [None, Some(Value::Null), Some(json!(""))] {
        let mut call = json!({"type":"function","function":{"name":"lookup","arguments":{}}});
        if let Some(id) = &id {
            call["id"] = id.clone();
        }
        let error = render(
            json!([
                {"role":"user","content":"lookup"},
                {"role":"assistant","content":"","tool_calls":[call]}
            ]),
            "high",
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("assistant tool_call is missing required 'id'")
        );

        let mut result = json!({"role":"tool","content":"42"});
        if let Some(id) = id {
            result["tool_call_id"] = id;
        }
        let error = render(
            json!([
                {"role":"user","content":"lookup"},
                {"role":"assistant","content":"","tool_calls":[
                    {"id":"call_a","type":"function","function":{"name":"lookup","arguments":{}}}
                ]}, result
            ]),
            "high",
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("tool message is missing required 'tool_call_id'")
        );
    }
}

#[test]
fn solar_valid_tool_roundtrip_preserves_reasoning() {
    let text = render(
        json!([
            {"role":"user","content":"lookup"},
            {"role":"assistant","content":"","reasoning_content":"PRIOR_REASONING","tool_calls":[
                {"id":"call_a","type":"function","function":{"name":"lookup","arguments":{}}}
            ]},
            {"role":"tool","tool_call_id":"call_a","content":"42"},
            {"role":"user","content":"continue"}
        ]),
        "xhigh",
    )
    .unwrap();
    assert!(text.contains("<|think:start|>PRIOR_REASONING<|think:end|>"));
    assert!(text.contains("<|tool_call:start|>lookup"));
    assert!(text.contains("42"));
    assert!(text.ends_with("<|think:start|>"));
}

#[test]
fn solar_effort_request_normalization_matches_template() {
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    for (top, nested, expected) in [
        (Some("xhigh"), "none", "xhigh"),
        (Some("none"), "xhigh", "none"),
        (None, "xhigh", "xhigh"),
    ] {
        let mut body = json!({"model":"solar","messages":[{"role":"user","content":"hello"}],
            "chat_template_kwargs":{"reasoning_effort":nested}});
        if let Some(top) = top {
            body["reasoning_effort"] = json!(top);
        }
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(body).unwrap();
        request.normalize_reasoning_template_args().unwrap();
        let effort = request.chat_template_args.as_ref().unwrap()["reasoning_effort"]
            .as_str()
            .unwrap();
        assert_eq!(effort, expected);
        let prompt = render(json!([{"role":"user","content":"hello"}]), effort).unwrap();
        assert_eq!(prompt.ends_with("<|think:start|>"), expected == "xhigh");
    }
}
