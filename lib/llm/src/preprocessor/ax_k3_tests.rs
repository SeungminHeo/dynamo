// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A.X-K3 (`ax_k3` unified family) request cases through the Dynamo frontend:
//! prompt-to-route handoff, tool_choice modes, finish reasons, usage, and both
//! streamed and collected responses. Parser-level grammar cases live with the
//! parser in the frontend-crates fork.

use super::*;
use crate::protocols::openai::chat_completions::aggregator::DeltaAggregator;
use crate::protocols::openai::chat_completions::unified_parser::{
    AX_K3_UNIFIED_FAMILY, configured_family, selected_family,
};
use serde_json::{Value, json};

/// The A.X-K3 generation prompt ends inside an opened thought.
const PROMPT: &str = "<|im_start|>user\n날씨?<|im_end|>\n<|im_start|>assistant\n<think>\n";
const CALL: &str = "<tool_call>get_weather<|parameter|>city\n서울<|/parameter|><|parameter|>days\n3<|/parameter|></tool_call>";

fn preprocessor(tool_call_parser: &str, reasoning_parser: &str) -> Arc<OpenAIPreprocessor> {
    let mut card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    card.runtime_config.reasoning_parser = Some(reasoning_parser.into());
    card.runtime_config.tool_call_parser = Some(tool_call_parser.into());
    OpenAIPreprocessor::new(card).unwrap()
}

fn ax_k3() -> Arc<OpenAIPreprocessor> {
    static PREPROCESSOR: OnceLock<Arc<OpenAIPreprocessor>> = OnceLock::new();
    PREPROCESSOR
        .get_or_init(|| preprocessor("ax_k3", "ax_k3"))
        .clone()
}

fn request(extra: Value) -> NvCreateChatCompletionRequest {
    let mut body = json!({
        "model": "ax-k3",
        "messages": [{"role": "user", "content": "날씨?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}
            }, "required": ["city"]}
        }}, {"type": "function", "function": {
            "name": "ping", "parameters": {"type": "object", "properties": {}}
        }}],
    });
    for (key, value) in extra.as_object().unwrap() {
        if value.is_null() {
            body.as_object_mut().unwrap().remove(key);
        } else {
            body[key] = value.clone();
        }
    }
    serde_json::from_value(body).unwrap()
}

fn chunk(
    text: Option<&str>,
    finish: Option<&str>,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    Annotated::from_data(
        serde_json::from_value(json!({
            "id": "ax", "object": "chat.completion.chunk", "created": 0, "model": "ax-k3",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
            "llm_metrics": {"input_tokens": 2, "output_tokens": 1, "chunk_tokens": 1}
        }))
        .unwrap(),
    )
}

fn usage_chunk(completion_tokens: u32) -> Annotated<NvCreateChatCompletionStreamResponse> {
    let mut usage = chunk(None, None);
    let data = usage.data.as_mut().unwrap();
    data.inner.choices.clear();
    data.llm_metrics = None;
    data.inner.usage = Some(
        serde_json::from_value(json!({
            "prompt_tokens": 2, "completion_tokens": completion_tokens,
            "total_tokens": 2 + completion_tokens
        }))
        .unwrap(),
    );
    usage
}

/// Text chunks, with the finish reason on a trailing empty chunk.
fn chunks(parts: &[&str], finish: &str) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    let mut out: Vec<_> = parts.iter().map(|part| chunk(Some(part), None)).collect();
    out.push(chunk(None, Some(finish)));
    out
}

fn prefill(prompt: &str) -> PromptReasoningPrefill {
    OpenAIPreprocessor::prompt_reasoning_prefill_for_parsers(
        Some("ax_k3"),
        Some("ax_k3"),
        Some(prompt),
    )
}

async fn stream_with(
    preprocessor: &OpenAIPreprocessor,
    req: &NvCreateChatCompletionRequest,
    prompt: &str,
    input: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
) -> Vec<Value> {
    preprocessor
        .postprocessor_parsing_stream(stream::iter(input), req, prefill(prompt), false)
        .unwrap()
        .map(|out| serde_json::to_value(out.data).unwrap())
        .collect()
        .await
}

async fn stream_out(
    req: &NvCreateChatCompletionRequest,
    input: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
) -> Vec<Value> {
    stream_with(&ax_k3(), req, PROMPT, input).await
}

async fn collected(
    req: &NvCreateChatCompletionRequest,
    input: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
) -> Value {
    let parsed = ax_k3()
        .postprocessor_parsing_stream(stream::iter(input), req, prefill(PROMPT), false)
        .unwrap();
    let response = DeltaAggregator::apply(
        parsed,
        crate::protocols::openai::ParsingOptions::new(Some("ax_k3".into()), Some("ax_k3".into())),
    )
    .await
    .unwrap();
    serde_json::to_value(response).unwrap()
}

fn choices(out: &[Value]) -> impl Iterator<Item = &Value> {
    out.iter()
        .flat_map(|v| v["choices"].as_array().into_iter().flatten())
}

fn text(out: &[Value], key: &str) -> String {
    choices(out)
        .filter_map(|c| c["delta"][key].as_str())
        .collect()
}

fn finish(out: &[Value]) -> Option<String> {
    choices(out)
        .filter_map(|c| c["finish_reason"].as_str())
        .last()
        .map(str::to_string)
}

/// Streamed tool calls merged by index: (id, name, parsed arguments).
fn calls(out: &[Value]) -> Vec<(String, String, Value)> {
    let mut merged: Vec<(String, String, String)> = Vec::new();
    for call in choices(out).flat_map(|c| c["delta"]["tool_calls"].as_array().into_iter().flatten())
    {
        let index = call["index"].as_u64().unwrap() as usize;
        if merged.len() <= index {
            merged.resize(index + 1, Default::default());
        }
        if let Some(id) = call["id"].as_str() {
            merged[index].0 = id.into();
        }
        if let Some(name) = call["function"]["name"].as_str() {
            merged[index].1.push_str(name);
        }
        if let Some(arguments) = call["function"]["arguments"].as_str() {
            merged[index].2.push_str(arguments);
        }
    }
    merged
        .into_iter()
        .map(|(id, name, arguments)| (id, name, serde_json::from_str(&arguments).unwrap()))
        .collect()
}

fn leaks_markup(s: &str) -> bool {
    [
        "<tool_call>",
        "</tool_call>",
        "<|parameter|>",
        "<|/parameter|>",
        "</think>",
    ]
    .iter()
    .any(|marker| s.contains(marker))
}

#[test]
fn ax_k3_parser_names_route_to_the_unified_family() {
    assert_eq!(
        configured_family(Some("ax_k3"), Some("ax_k3")),
        Some(AX_K3_UNIFIED_FAMILY)
    );
    for alias in ["axk3", "ax-k3"] {
        assert_eq!(
            configured_family(Some(alias), Some(alias)),
            Some(AX_K3_UNIFIED_FAMILY)
        );
    }
    // Existing deployments pair the A.X-K3 tool grammar with `deepseek_r1`.
    assert_eq!(
        configured_family(Some("ax_k3"), Some("deepseek_r1")),
        Some(AX_K3_UNIFIED_FAMILY)
    );
    assert_eq!(
        configured_family(None, Some("ax_k3")),
        Some(AX_K3_UNIFIED_FAMILY)
    );
    assert_eq!(
        selected_family(Some("ax_k3"), Some("ax_k3")),
        Some(AX_K3_UNIFIED_FAMILY)
    );
    assert_eq!(configured_family(Some("ax_k3"), Some("qwen3")), None);
    assert!(
        OpenAIPreprocessor::parser_requires_special_tokens(Some("ax_k3"), Some("ax_k3")).unwrap()
    );
}

#[test]
fn ax_k3_prompt_prefill_reads_the_rendered_thought() {
    assert!(prefill(PROMPT).unified);
    assert!(!prefill(PROMPT).response_start);
    let closed = prefill("<|im_start|>assistant\n<think></think>\n");
    assert!(!closed.unified);
    assert!(closed.response_start);
}

#[test]
fn ax_k3_thinking_cannot_be_turned_off_by_request() {
    for extra in [
        json!({"chat_template_kwargs": {"enable_thinking": false}}),
        json!({"reasoning_effort": "low"}),
        json!({"chat_template_kwargs": {"reasoning_effort": "max"}}),
    ] {
        let mut req = request(extra.clone());
        req.normalize_reasoning_template_args().unwrap();
        assert!(
            !OpenAIPreprocessor::request_disables_reasoning_with_family(
                &req,
                Some("ax_k3"),
                Some("ax_k3"),
                None,
                Some(AX_K3_UNIFIED_FAMILY),
            ),
            "{extra}"
        );
    }
}

#[tokio::test]
async fn ax_k3_auto_reasoning_content_and_call_at_every_utf8_split() {
    let req = request(json!({"tool_choice": "auto"}));
    let raw = format!("날씨를 조회하자.\n</think>\n\n확인할게요. {CALL}");
    for split in raw.char_indices().map(|(i, _)| i).chain([raw.len()]) {
        let out = stream_out(&req, chunks(&[&raw[..split], &raw[split..]], "stop")).await;
        assert_eq!(
            text(&out, "reasoning_content"),
            "날씨를 조회하자.\n",
            "split {split}"
        );
        assert_eq!(text(&out, "content").trim(), "확인할게요.", "split {split}");
        let calls = calls(&out);
        assert_eq!(calls.len(), 1, "split {split}: {out:?}");
        assert_eq!(calls[0].1, "get_weather");
        assert_eq!(calls[0].2, json!({"city": "서울", "days": 3}));
        assert!(!calls[0].0.is_empty());
        assert_eq!(finish(&out).as_deref(), Some("tool_calls"), "split {split}");
    }
}

#[tokio::test]
async fn ax_k3_parallel_calls_keep_ids_indices_and_verbatim_arguments_per_character() {
    let req = request(json!({"tool_choice": "auto"}));
    let raw = format!("r</think>확인할게요.\n{CALL}\n<tool_call>ping</tool_call>\n");
    let parts: Vec<String> = raw.chars().map(String::from).collect();
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    let out = stream_out(&req, chunks(&parts, "stop")).await;
    let calls = calls(&out);
    assert_eq!(calls.len(), 2, "{out:?}");
    assert_ne!(calls[0].0, calls[1].0);
    assert_eq!(
        (
            calls[0].1.as_str(),
            &calls[0].2,
            calls[1].1.as_str(),
            &calls[1].2
        ),
        (
            "get_weather",
            &json!({"city": "서울", "days": 3}),
            "ping",
            &json!({})
        )
    );
    // Reference argument text: `json.dumps` separators, values in model spelling.
    let raw_arguments: String = choices(&out)
        .flat_map(|c| c["delta"]["tool_calls"].as_array().into_iter().flatten())
        .filter(|call| call["index"] == 0)
        .filter_map(|call| call["function"]["arguments"].as_str())
        .collect();
    assert_eq!(raw_arguments, r#"{"city": "서울", "days": 3}"#);
    assert_eq!(text(&out, "content").trim(), "확인할게요.");
    assert_eq!(finish(&out).as_deref(), Some("tool_calls"));
}

#[tokio::test]
async fn ax_k3_every_call_runs_and_is_sent_when_it_closes() {
    // `call_group_policy = "all"`: every complete call runs, the text between
    // them is content, and a call is sent as soon as its `</tool_call>`
    // arrives (before the text that follows it).
    let req = request(json!({"tool_choice": "auto"}));
    let ping = "<tool_call>ping</tool_call>";
    let out = stream_out(
        &req,
        chunks(
            &[
                &format!("r</think>먼저 {CALL}"),
                " 잠깐, 다시. ",
                ping,
                " 끝.",
            ],
            "stop",
        ),
    )
    .await;
    let calls = calls(&out);
    let names: Vec<&str> = calls.iter().map(|call| call.1.as_str()).collect();
    assert_eq!(names, ["get_weather", "ping"], "{out:?}");
    assert_eq!(calls[0].2, json!({"city": "서울", "days": 3}));
    assert_ne!(calls[0].0, calls[1].0);
    assert_eq!(text(&out, "content"), "먼저  잠깐, 다시.  끝.");
    assert_eq!(finish(&out).as_deref(), Some("tool_calls"));
    let position = |found: &dyn Fn(&Value) -> bool| {
        out.iter()
            .position(|chunk| choices(std::slice::from_ref(chunk)).any(found))
            .unwrap()
    };
    let first_call = position(&|choice| choice["delta"]["tool_calls"].is_array());
    let later_text = position(&|choice| {
        choice["delta"]["content"]
            .as_str()
            .is_some_and(|content| content.contains("잠깐"))
    });
    assert!(first_call < later_text, "{out:?}");
}

#[tokio::test]
async fn ax_k3_collected_response_preserves_reasoning_calls_and_finish() {
    let req = request(json!({"tool_choice": "auto"}));
    let value = collected(&req, chunks(&["plan</think>", CALL], "stop")).await;
    let choice = &value["choices"][0];
    assert_eq!(choice["message"]["reasoning_content"], "plan");
    assert_eq!(choice["finish_reason"], "tool_calls");
    let calls = choice["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"city": "서울", "days": 3})
    );
}

#[tokio::test]
async fn ax_k3_plain_answer_without_tools() {
    let req = request(json!({"tools": null}));
    let out = stream_out(&req, chunks(&["생각</th", "ink>\n\n안녕하세요!"], "stop")).await;
    assert_eq!(text(&out, "reasoning_content"), "생각");
    assert_eq!(text(&out, "content"), "\n\n안녕하세요!");
    assert_eq!(finish(&out).as_deref(), Some("stop"));
}

#[tokio::test]
async fn ax_k3_reasoning_cut_by_length_stays_reasoning() {
    let req = request(json!({"tool_choice": "auto"}));
    let out = stream_out(&req, chunks(&["아직 생각 중"], "length")).await;
    assert_eq!(text(&out, "reasoning_content"), "아직 생각 중");
    assert!(text(&out, "content").is_empty());
    assert_eq!(finish(&out).as_deref(), Some("length"));
}

#[tokio::test]
async fn ax_k3_truncated_call_is_returned_as_content() {
    // Reference: a call without `</tool_call>` is never executed; its text is content.
    let req = request(json!({"tool_choice": "auto"}));
    let open = "<tool_call>get_weather<|parameter|>city\n서";
    let out = stream_out(&req, chunks(&["r</think>", open], "length")).await;
    assert!(calls(&out).is_empty(), "{out:?}");
    assert_eq!(text(&out, "content"), open);
    assert_eq!(finish(&out).as_deref(), Some("length"));
}

#[tokio::test]
async fn ax_k3_completed_call_survives_a_truncated_follower() {
    let req = request(json!({"tool_choice": "auto"}));
    let open = "<tool_call>ping<|parameter|>x\n";
    let out = stream_out(&req, chunks(&["r</think>", CALL, open], "length")).await;
    let calls = calls(&out);
    assert_eq!(calls.len(), 1, "{out:?}");
    assert_eq!(calls[0].1, "get_weather");
    assert_eq!(text(&out, "content"), open);
    // The completed call is delivered; the backend truncation stays visible.
    assert_eq!(finish(&out).as_deref(), Some("length"));
}

#[tokio::test]
async fn ax_k3_marker_text_inside_a_string_argument_is_data() {
    let req = request(json!({"tool_choice": "auto"}));
    let raw = "r</think><tool_call>get_weather<|parameter|>city\n<tool_call>x</tool_call> </think> 서울<|/parameter|></tool_call>";
    let out = stream_out(&req, chunks(&[raw], "stop")).await;
    let calls = calls(&out);
    assert_eq!(calls.len(), 1, "{out:?}");
    assert_eq!(
        calls[0].2,
        json!({"city": "<tool_call>x</tool_call> </think> 서울"})
    );
}

#[tokio::test]
async fn ax_k3_tool_choice_none_returns_no_calls() {
    let req = request(json!({"tool_choice": "none"}));
    let out = stream_out(&req, chunks(&["r</think>답변입니다."], "stop")).await;
    assert!(calls(&out).is_empty());
    assert_eq!(text(&out, "reasoning_content"), "r");
    assert_eq!(text(&out, "content"), "답변입니다.");
    assert_eq!(finish(&out).as_deref(), Some("stop"));
    // A native call the model emits anyway is never executable under `none`.
    let out = stream_out(&req, chunks(&["r</think>", CALL], "stop")).await;
    assert!(calls(&out).is_empty(), "{out:?}");
    assert_eq!(finish(&out).as_deref(), Some("stop"));
    assert!(!leaks_markup(&text(&out, "content")), "{out:?}");
}

#[tokio::test]
async fn ax_k3_required_accepts_guided_json_with_and_without_reasoning() {
    let req = request(json!({"tool_choice": "required"}));
    let payload = r#"[{"name":"get_weather","parameters":{"city":"서울","days":3}}]"#;
    for (prefix, reasoning) in [("", ""), ("계획</think>", "계획")] {
        let raw = format!("{prefix}{payload}");
        for parts in [
            vec![raw.as_str()],
            raw.split_inclusive(['>', ',']).collect(),
        ] {
            let out = stream_out(&req, chunks(&parts, "stop")).await;
            assert_eq!(text(&out, "reasoning_content"), reasoning, "{out:?}");
            let calls = calls(&out);
            assert_eq!(calls.len(), 1, "{prefix:?} {out:?}");
            assert_eq!(calls[0].2, json!({"city": "서울", "days": 3}));
            assert!(text(&out, "content").is_empty(), "{out:?}");
            assert_eq!(finish(&out).as_deref(), Some("tool_calls"));
        }
    }
}

#[tokio::test]
async fn ax_k3_named_choice_accepts_bare_guided_arguments() {
    let req =
        request(json!({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}));
    for raw in [
        r#"{"city":"서울","days":3}"#,
        r#"plan</think>{"city":"서울","days":3}"#,
    ] {
        let out = stream_out(&req, chunks(&[raw], "stop")).await;
        let calls = calls(&out);
        assert_eq!(calls.len(), 1, "{raw}: {out:?}");
        assert_eq!(calls[0].1, "get_weather");
        assert_eq!(calls[0].2, json!({"city": "서울", "days": 3}));
        assert!(text(&out, "content").is_empty(), "{out:?}");
    }
}

#[tokio::test]
async fn ax_k3_collected_matrix_preserves_channels_and_finish() {
    let auto = request(json!({"tool_choice": "auto"}));
    let required = request(json!({"tool_choice": "required"}));
    let guided = r#"[{"name":"get_weather","parameters":{"city":"서울","days":3}}]"#;
    struct Case<'a> {
        name: &'a str,
        request: &'a NvCreateChatCompletionRequest,
        parts: Vec<&'a str>,
        backend_finish: &'a str,
        reasoning: &'a str,
        content: Option<&'a str>,
        calls: usize,
        finish: &'a str,
    }
    let case = |name, request, parts, backend_finish, reasoning, content, calls, finish| Case {
        name,
        request,
        parts,
        backend_finish,
        reasoning,
        content,
        calls,
        finish,
    };
    let cases = [
        case(
            "reasoning+call",
            &auto,
            vec!["plan</think>", CALL],
            "stop",
            "plan",
            None,
            1,
            "tool_calls",
        ),
        case(
            "reasoning+answer",
            &auto,
            vec!["plan</think>", "답변"],
            "stop",
            "plan",
            Some("답변"),
            0,
            "stop",
        ),
        case(
            "required guided",
            &required,
            vec![guided],
            "stop",
            "",
            None,
            1,
            "tool_calls",
        ),
        case(
            "required after reasoning",
            &required,
            vec!["plan</think>", guided],
            "stop",
            "plan",
            None,
            1,
            "tool_calls",
        ),
        case(
            "reasoning cut",
            &auto,
            vec!["아직 생각"],
            "length",
            "아직 생각",
            None,
            0,
            "length",
        ),
    ];
    for Case {
        name,
        request: req,
        parts,
        backend_finish: backend,
        reasoning,
        content,
        calls: n_calls,
        finish,
    } in cases
    {
        let value = collected(req, chunks(&parts, backend)).await;
        let choice = &value["choices"][0];
        let message = &choice["message"];
        assert_eq!(
            message["reasoning_content"].as_str().unwrap_or_default(),
            reasoning,
            "{name}: {value}"
        );
        assert_eq!(
            message["content"].as_str().filter(|text| !text.is_empty()),
            content,
            "{name}: {value}"
        );
        assert_eq!(
            message["tool_calls"].as_array().map_or(0, Vec::len),
            n_calls,
            "{name}: {value}"
        );
        assert_eq!(choice["finish_reason"], finish, "{name}: {value}");
    }
}

#[tokio::test]
async fn ax_k3_call_inside_an_unclosed_thought_stays_reasoning() {
    // Reference (vLLM `axk3` / DeepSeek-R1): only `</think>` ends a thought, so a
    // call written while still thinking is never dispatched.
    let req = request(json!({"tool_choice": "auto"}));
    let out = stream_out(
        &req,
        chunks(&["계획: ", CALL, " 그리고</think>답변"], "stop"),
    )
    .await;
    assert!(calls(&out).is_empty(), "{out:?}");
    assert_eq!(
        text(&out, "reasoning_content"),
        format!("계획: {CALL} 그리고")
    );
    assert_eq!(text(&out, "content"), "답변");
    assert_eq!(finish(&out).as_deref(), Some("stop"));
}

#[test]
fn ax_k3_top_level_reasoning_effort_reaches_the_template() {
    for effort in ["low", "high", "max"] {
        let mut req = request(json!({"reasoning_effort": effort}));
        req.normalize_reasoning_template_args().unwrap();
        assert_eq!(
            req.chat_template_args
                .as_ref()
                .and_then(|args| args.get("reasoning_effort")),
            Some(&json!(effort)),
            "{effort}"
        );
    }
}

#[tokio::test]
async fn ax_k3_response_format_json_from_the_first_token() {
    let req = request(json!({"tools": null, "response_format": {"type": "json_object"}}));
    let out = stream_out(&req, chunks(&[r#"{"answer":"맑음"}"#], "stop")).await;
    assert!(text(&out, "reasoning_content").is_empty(), "{out:?}");
    assert_eq!(
        serde_json::from_str::<Value>(&text(&out, "content")).unwrap(),
        json!({"answer": "맑음"})
    );
}

#[tokio::test]
async fn ax_k3_response_format_json_follows_reasoning() {
    let req = request(json!({"tools": null, "response_format": {"type": "json_object"}}));
    let out = stream_out(
        &req,
        chunks(&["plan</think>", r#"{"answer":"맑음"}"#], "stop"),
    )
    .await;
    assert_eq!(text(&out, "reasoning_content"), "plan");
    assert_eq!(
        serde_json::from_str::<Value>(&text(&out, "content")).unwrap(),
        json!({"answer": "맑음"})
    );
}

#[tokio::test]
async fn ax_k3_closed_thought_prompt_starts_in_the_answer() {
    let req = request(json!({"tool_choice": "auto"}));
    let out = stream_with(
        &ax_k3(),
        &req,
        "<|im_start|>assistant\n<think></think>\n",
        chunks(&["바로 답변 ", CALL], "stop"),
    )
    .await;
    assert!(text(&out, "reasoning_content").is_empty(), "{out:?}");
    assert_eq!(text(&out, "content").trim(), "바로 답변");
    assert_eq!(calls(&out).len(), 1);
}

#[tokio::test]
async fn ax_k3_deepseek_r1_reasoning_pairing_matches_the_unified_family() {
    let legacy_pair = preprocessor("ax_k3", "deepseek_r1");
    let req = request(json!({"tool_choice": "auto"}));
    let out = stream_with(
        &legacy_pair,
        &req,
        PROMPT,
        chunks(&["plan</think>", CALL], "stop"),
    )
    .await;
    assert_eq!(text(&out, "reasoning_content"), "plan");
    assert_eq!(calls(&out).len(), 1, "{out:?}");
}

#[tokio::test]
async fn ax_k3_reasoning_tokens_are_reported() {
    let req = request(json!({"tool_choice": "auto"}));
    let mut input = vec![
        chunk(Some("plan"), None),
        chunk(Some("</think>answer"), Some("stop")),
    ];
    input.push(usage_chunk(2));
    let out = stream_out(&req, input).await;
    assert_eq!(text(&out, "reasoning_content"), "plan");
    assert_eq!(text(&out, "content"), "answer");
    assert_eq!(
        out.last().unwrap()["usage"]["completion_tokens_details"]["reasoning_tokens"],
        1
    );
}
