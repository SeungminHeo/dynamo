// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Solar calls are committed only when the backend finishes normally. A later
//! length limit, cancellation, error or EOF invalidates even earlier complete
//! calls. Keep reasoning/prose streaming; only executable deltas are held.
use super::*;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_protocols::types::{ChatCompletionMessageToolCallChunk, FinishReason};

/// Cancellation becomes OpenAI `stop` in the generic delta generator. Reject
/// it before that lossy projection, or terminal-only calls could be committed.
pub(super) fn reject_cancelled(
    mut response: Annotated<BackendOutput>,
    trace: Option<&crate::request_trace::SharedFinishReasonMetadata>,
) -> Annotated<BackendOutput> {
    if let Some(output) = &response.data
        && matches!(
            output.finish_reason,
            Some(crate::protocols::common::FinishReason::Cancelled)
        )
    {
        crate::request_trace::record_backend_finish_reason_metadata(
            trace,
            output.index,
            output.finish_reason.as_ref(),
            output.stop_reason.as_ref(),
        );
        response.data = None;
        response.event = Some("error".into());
        response.error = Some(
            DynamoError::builder()
                .error_type(ErrorType::Cancelled)
                .message("backend cancelled Solar tool generation")
                .build(),
        );
    }
    response
}

#[derive(Default)]
struct Completion {
    finishes: HashMap<u32, FinishReason>,
    failed: bool,
}

pub(super) fn apply_tool_policy<S>(
    stream: S,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    tools: Option<Vec<ToolDefinition>>,
    uses_structural_tag: bool,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    // The jail may synthesize a tool_calls finish at EOF. Remember only actual
    // backend terminal reasons, before the jail, so synthesis cannot commit a
    // cancelled request. Both stream closures share state and must remain Send.
    // The lock is touched only on terminal/error chunks.
    let completion = Arc::new(parking_lot::Mutex::new(Completion::default()));
    let input_completion = completion.clone();
    let stream = stream.map(move |response| {
        if response.error.is_some() || response.event.as_deref() == Some("error") {
            input_completion.lock().failed = true;
        }
        if let Some(data) = &response.data {
            for choice in &data.inner.choices {
                if let Some(reason) = choice.finish_reason {
                    input_completion
                        .lock()
                        .finishes
                        .insert(choice.index, reason);
                }
            }
        }
        response
    });
    let parsed = OpenAIPreprocessor::apply_tool_calling_jail(
        Some("solar_open2".into()),
        tool_choice,
        tools,
        uses_structural_tag,
        false, // No speculative guided deltas: a later failure must revoke all calls.
        stream,
    );
    let mut pending: HashMap<u32, Vec<ChatCompletionMessageToolCallChunk>> = HashMap::new();
    let mut tool_section = HashSet::new();
    parsed.map(move |mut response| {
        if let Some(data) = response.data.as_mut() {
            for choice in &mut data.inner.choices {
                if tool_section.contains(&choice.index) {
                    choice.delta.content = None;
                }
                if let Some(calls) = choice.delta.tool_calls.take() {
                    tool_section.insert(choice.index);
                    pending.entry(choice.index).or_default().extend(calls);
                }
                if choice.finish_reason.is_some() {
                    let state = completion.lock();
                    let backend_finish = state.finishes.get(&choice.index).copied();
                    let commit = !state.failed && backend_finish == Some(FinishReason::Stop);
                    drop(state);
                    if let Some(calls) = pending
                        .remove(&choice.index)
                        .filter(|calls| !calls.is_empty())
                    {
                        if commit {
                            choice.delta.tool_calls = Some(calls);
                            choice.finish_reason = Some(FinishReason::ToolCalls);
                        } else {
                            // Preserve actual truncation instead of the jail's synthetic
                            // tool_calls. Missing terminal is an interrupted generation.
                            choice.finish_reason =
                                Some(backend_finish.unwrap_or(FinishReason::Length));
                        }
                    } else if !commit && choice.finish_reason == Some(FinishReason::ToolCalls) {
                        choice.finish_reason = Some(backend_finish.unwrap_or(FinishReason::Length));
                    }
                }
            }
        }
        response
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::chat_completions::{
        NvCreateChatCompletionResponse, aggregator::ChatCompletionAggregator,
    };
    use serde_json::{Value, json};

    fn preprocessor() -> Arc<OpenAIPreprocessor> {
        static PREPROCESSOR: OnceLock<Arc<OpenAIPreprocessor>> = OnceLock::new();
        PREPROCESSOR
            .get_or_init(|| {
                let mut card = ModelDeploymentCard::load_from_disk(
                    "tests/data/sample-models/mock-llama-3.1-8b-instruct",
                    None,
                )
                .unwrap();
                card.runtime_config.reasoning_parser = Some("solar_open2".into());
                card.runtime_config.tool_call_parser = Some("solar_open2".into());
                OpenAIPreprocessor::new(card).unwrap()
            })
            .clone()
    }

    fn request(effort: Option<&str>, choice: Value) -> NvCreateChatCompletionRequest {
        let mut body = json!({"model":"solar", "messages":[{"role":"user","content":"hello"}],
            "tools":[{"type":"function","function":{"name":"search","parameters":{
                "type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}}
            }}}], "tool_choice": choice});
        if let Some(effort) = effort {
            body["reasoning_effort"] = json!(effort);
        }
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(body).unwrap();
        request.normalize_reasoning_template_args().unwrap();
        request
    }

    fn chunk(
        text: Option<&str>,
        finish: Option<&str>,
        index: u32,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        Annotated::from_data(
            serde_json::from_value(json!({
                "id":"solar", "object":"chat.completion.chunk", "created":0, "model":"solar",
                "choices":[{"index":index,"delta":{"content":text},"finish_reason":finish}],
                "llm_metrics":{"input_tokens":2,"output_tokens":1,"chunk_tokens":1}
            }))
            .unwrap(),
        )
    }

    async fn run(
        req: &NvCreateChatCompletionRequest,
        chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
    ) -> Vec<Value> {
        preprocessor()
            .postprocessor_parsing_stream(stream::iter(chunks), req, false, false)
            .unwrap()
            .map(|out| serde_json::to_value(out.data).unwrap())
            .collect()
            .await
    }

    fn text(out: &[Value], key: &str) -> String {
        out.iter()
            .flat_map(|v| v["choices"].as_array().into_iter().flatten())
            .filter_map(|c| c["delta"][key].as_str())
            .collect()
    }

    fn calls(out: &[Value]) -> Vec<Value> {
        out.iter()
            .flat_map(|v| v["choices"].as_array().into_iter().flatten())
            .flat_map(|c| {
                c["delta"]["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect()
    }

    const NATIVE: &str = "<|tool_call:start|>search\n<|tool_arg:start|>query<|tool_arg:value|>서울<|tool_arg:end|>\n<|tool_arg:start|>limit<|tool_arg:value|>7<|tool_arg:end|>\n<|tool_call:end|>";

    #[tokio::test]
    async fn solar_collected_responses_preserve_terminal_and_tool_policy() {
        for finish in ["stop", "length", "repetition"] {
            for tool_choice in [
                json!("auto"),
                json!("required"),
                json!({"type":"function","function":{"name":"search"}}),
            ] {
                let req = request(Some("high"), tool_choice.clone());
                let body = if tool_choice == json!("auto") {
                    format!("plan<|think:end|>{NATIVE}")
                } else if tool_choice == json!("required") {
                    r#"plan<|think:end|>[{"name":"search","parameters":{"query":"서울","limit":7}}]"#.into()
                } else {
                    r#"plan<|think:end|>{"query":"서울","limit":7}"#.into()
                };
                let parsed = preprocessor()
                    .postprocessor_parsing_stream(
                        stream::iter(vec![chunk(Some(&body), Some(finish), 0)]),
                        &req,
                        false,
                        false,
                    )
                    .unwrap();
                let options = crate::protocols::openai::ParsingOptions::new(
                    Some("solar_open2".into()),
                    Some("solar_open2".into()),
                );
                let response =
                    NvCreateChatCompletionResponse::from_annotated_stream(parsed, options)
                        .await
                        .unwrap();
                let value = serde_json::to_value(response).unwrap();
                let choice = &value["choices"][0];
                assert_eq!(
                    choice["message"]["reasoning_content"], "plan",
                    "{finish}/{tool_choice}"
                );
                if finish == "stop" {
                    assert_eq!(choice["finish_reason"], "tool_calls");
                    assert_eq!(choice["message"]["tool_calls"].as_array().unwrap().len(), 1);
                } else {
                    assert_eq!(choice["finish_reason"], finish);
                    assert!(choice["message"]["tool_calls"].is_null());
                }
            }
        }
    }

    #[test]
    fn solar_effort_template_and_backend_gate() {
        for (effort, enabled) in [
            (None, true),
            (Some("high"), true),
            (Some("medium"), true),
            (Some("low"), false),
            (Some("none"), false),
            (Some("minimal"), false),
            (Some("xhigh"), false),
        ] {
            let req = request(effort, json!("required"));
            assert_eq!(
                OpenAIPreprocessor::is_reasoning_disabled_by_request(
                    Some("solar_open2"),
                    req.chat_template_args.as_ref()
                ),
                !enabled,
                "{effort:?}"
            );
            assert_eq!(
                OpenAIPreprocessor::guided_output_requires_reasoning(&req, Some("solar_open2")),
                enabled,
                "{effort:?}"
            );
        }
        let mut req: NvCreateChatCompletionRequest = serde_json::from_value(json!({"model":"solar","messages":[], "reasoning_effort":"high", "chat_template_kwargs":{"reasoning_effort":"low","enable_thinking":false}})).unwrap();
        req.normalize_reasoning_template_args().unwrap();
        assert!(!OpenAIPreprocessor::is_reasoning_disabled_by_request(
            Some("solar_open2"),
            req.chat_template_args.as_ref()
        ));
        let args = HashMap::from([("enable_thinking".into(), json!(false))]);
        assert!(!OpenAIPreprocessor::is_reasoning_disabled_by_request(
            Some("solar_open2"),
            Some(&args)
        ));
        assert!(OpenAIPreprocessor::prompt_injected_reasoning_start(
            Some("solar_open2"),
            Some("...<|think:start|>")
        ));
        assert!(!OpenAIPreprocessor::prompt_injected_reasoning_start(
            Some("solar_open2"),
            Some("...<|think:start|><|think:end|>")
        ));
    }

    #[test]
    fn solar_template_field_contract() {
        let template = "{{ tools[0].function | tojson }}|{% for m in messages %}{% for t in m.tool_calls|default([]) %}{% for k,v in t.function.arguments|items %}{{ k }}={{ v|tojson }};{% endfor %}{% endfor %}{% endfor %}|{{ response_format|default(none)|tojson }}";
        let PromptFormatter::OAI(formatter) = PromptFormatter::from_parts(
            serde_json::from_value(json!({"chat_template":template})).unwrap(),
            dynamo_renderer::ContextMixins::default(),
            false,
        )
        .unwrap();
        let mut req: NvCreateChatCompletionRequest = serde_json::from_str(r#"{
            "model":"solar", "tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}}}}}],
            "messages":[{"role":"user","content":"hi"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{\"z\":\"서울\",\"a\":{\"b\":1,\"a\":[true,null]}}"}}]}],
            "response_format":{"type":"json_object"}
        }"#).unwrap();
        let rendered = formatter.render(&req).unwrap();
        assert!(rendered.starts_with(r#"{"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "number"}}}}|"#), "{rendered}");
        assert!(
            rendered.ends_with(r#"z="서울";a={"b": 1, "a": [true, null]};|null"#),
            "{rendered}"
        );
        req.chat_template_args = Some(HashMap::from([(
            "response_format".into(),
            json!({"type":"json_object"}),
        )]));
        assert!(
            formatter
                .render(&req)
                .unwrap()
                .ends_with(r#"|{"type": "json_object"}"#)
        );
    }

    #[tokio::test]
    async fn solar_low_none_content_and_usage() {
        for effort in ["low", "none"] {
            let mut usage = chunk(None, None, 0);
            let data = usage.data.as_mut().unwrap();
            data.inner.choices.clear();
            data.inner.usage = Some(
                serde_json::from_value(
                    json!({"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}),
                )
                .unwrap(),
            );
            let out = run(
                &request(Some(effort), json!("auto")),
                vec![chunk(Some("안녕"), Some("stop"), 0), usage],
            )
            .await;
            assert_eq!(text(&out, "content"), "안녕");
            assert!(text(&out, "reasoning_content").is_empty());
            assert_eq!(
                out.last().unwrap()["usage"]["completion_tokens_details"]["reasoning_tokens"],
                0
            );
        }
    }

    #[tokio::test]
    async fn solar_native_every_utf8_split() {
        let raw = format!("생각<|think:end|>{NATIVE}");
        for split in raw
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(raw.len()))
        {
            let out = run(
                &request(None, json!("auto")),
                vec![
                    chunk(Some(&raw[..split]), None, 0),
                    chunk(Some(&raw[split..]), Some("stop"), 0),
                ],
            )
            .await;
            assert_eq!(text(&out, "reasoning_content"), "생각", "split {split}");
            assert!(text(&out, "content").is_empty(), "split {split}: {out:?}");
            let calls = calls(&out);
            assert_eq!(calls.len(), 1, "split {split}: {out:?}");
            assert_eq!(
                serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap())
                    .unwrap(),
                json!({"query":"서울","limit":7})
            );
        }
    }

    #[tokio::test]
    async fn solar_guided_single_chunk_preserves_reasoning_and_bare_json() {
        for (choice, payload) in [
            (
                json!("required"),
                r#"[{"name":"search","parameters":{"query":"서울"}}]"#,
            ),
            (
                json!({"type":"function","function":{"name":"search"}}),
                r#"{"query":"서울"}"#,
            ),
        ] {
            for prefix in ["", "계획<|think:end|>"] {
                let raw = format!("{prefix}{payload}");
                for parts in [vec![raw.as_str()], raw.split_inclusive('>').collect()] {
                    let mut input = parts
                        .iter()
                        .map(|part| chunk(Some(part), None, 0))
                        .collect::<Vec<_>>();
                    input
                        .last_mut()
                        .unwrap()
                        .data
                        .as_mut()
                        .unwrap()
                        .inner
                        .choices[0]
                        .finish_reason = Some(FinishReason::Stop);
                    let out = run(&request(None, choice.clone()), input).await;
                    assert_eq!(
                        text(&out, "reasoning_content"),
                        if prefix.is_empty() { "" } else { "계획" },
                        "{out:?}"
                    );
                    assert_eq!(calls(&out).len(), 1, "{out:?}");
                    assert!(text(&out, "content").is_empty(), "{out:?}");
                }
            }
        }
    }

    #[tokio::test]
    async fn solar_truncation_discards_all_calls_including_completed_prefix() {
        for choice in [
            json!("auto"),
            json!("required"),
            json!({"type":"function","function":{"name":"search"}}),
        ] {
            let raw = if choice == "auto" {
                NATIVE
            } else if choice == "required" {
                r#"[{"name":"search","parameters":{"query":"x"}}]"#
            } else {
                r#"{"query":"x"}"#
            };
            for finish in [
                Some("length"),
                Some("content_filter"),
                Some("repetition"),
                None,
            ] {
                let out = run(
                    &request(Some("low"), choice.clone()),
                    vec![
                        chunk(Some(raw), None, 0),
                        chunk(Some("<|tool_call:start|>cut"), finish, 0),
                    ],
                )
                .await;
                assert!(calls(&out).is_empty(), "{choice} {finish:?}: {out:?}");
                assert!(!out.iter().any(|v| {
                    v["choices"]
                        .as_array()
                        .is_some_and(|cs| cs.iter().any(|c| c["finish_reason"] == "tool_calls"))
                }));
            }
        }
    }

    #[tokio::test]
    async fn solar_no_calls_before_backend_stop_and_choice_isolation() {
        let tools = Some(vec![ToolDefinition {
            name: "search".into(),
            parameters: None,
            strict: None,
        }]);
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let input = tokio_stream::wrappers::ReceiverStream::new(rx);
        let output = apply_tool_policy(
            input,
            Some(ChatCompletionToolChoiceOption::Auto),
            tools,
            false,
        );
        tokio::pin!(output);
        tx.send(chunk(Some(NATIVE), None, 0)).await.unwrap();
        // Any immediately available envelopes must carry no executable delta.
        while let Ok(Some(out)) =
            tokio::time::timeout(std::time::Duration::from_millis(10), output.next()).await
        {
            assert!(
                out.data
                    .unwrap()
                    .inner
                    .choices
                    .iter()
                    .all(|c| c.delta.tool_calls.is_none())
            );
        }
        tx.send(chunk(Some(NATIVE), Some("length"), 1))
            .await
            .unwrap();
        tx.send(chunk(None, Some("stop"), 0)).await.unwrap();
        drop(tx);
        let out: Vec<_> = output
            .map(|out| serde_json::to_value(out.data).unwrap())
            .collect()
            .await;
        assert_eq!(calls(&out).len(), 1, "{out:?}");
        assert!(
            out.iter()
                .flat_map(|v| v["choices"].as_array().into_iter().flatten())
                .filter(|c| c["index"] == 1)
                .all(|c| c["delta"]["tool_calls"].is_null())
        );
    }

    #[tokio::test]
    async fn solar_backend_cancellation_is_not_a_successful_stop() {
        let output: BackendOutput = serde_json::from_value(json!({
            "token_ids":[], "tokens":[], "text":NATIVE, "finish_reason":"cancelled"
        }))
        .unwrap();
        let req = request(Some("low"), json!("auto"));
        let mut generator = req.response_generator("cancelled".into());
        // Establish why gating after the generic projection alone is unsafe.
        assert_eq!(
            generator
                .choice_from_postprocessor(output.clone())
                .unwrap()
                .inner
                .choices[0]
                .finish_reason,
            Some(FinishReason::Stop)
        );
        let guarded = reject_cancelled(Annotated::from_data(output), None);
        assert!(guarded.data.is_none());
        assert!(guarded.error.is_some());
        let guarded = guarded.map_data(|output| {
            generator
                .choice_from_postprocessor(output)
                .map_err(|e| e.to_string())
        });
        let out = run(&req, vec![chunk(Some(NATIVE), None, 0), guarded]).await;
        assert!(calls(&out).is_empty(), "{out:?}");
    }

    #[tokio::test]
    async fn solar_backend_error_never_commits_calls() {
        let input = vec![
            chunk(Some(NATIVE), None, 0),
            Annotated::from_error("backend failed"),
        ];
        let out = run(&request(Some("low"), json!("auto")), input).await;
        assert!(calls(&out).is_empty(), "{out:?}");
    }

    #[tokio::test]
    async fn solar_reasoning_usage_is_populated() {
        let mut usage = chunk(None, None, 0);
        let data = usage.data.as_mut().unwrap();
        data.inner.choices.clear();
        data.llm_metrics = None;
        data.inner.usage = Some(
            serde_json::from_value(
                json!({"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}),
            )
            .unwrap(),
        );
        let out = run(
            &request(Some("high"), json!("auto")),
            vec![
                chunk(Some("plan"), None, 0),
                chunk(Some("<|think:end|>answer"), Some("stop"), 0),
                usage,
            ],
        )
        .await;
        assert_eq!(text(&out, "reasoning_content"), "plan");
        assert_eq!(text(&out, "content"), "answer");
        assert_eq!(
            out.last().unwrap()["usage"]["completion_tokens_details"]["reasoning_tokens"],
            1
        );
    }

    #[tokio::test]
    async fn solar_tool_escape_and_tool_choice_none() {
        let out = run(
            &request(None, json!("auto")),
            vec![chunk(Some(&format!("plan {NATIVE}")), Some("stop"), 0)],
        )
        .await;
        assert_eq!(text(&out, "reasoning_content"), "plan ");
        assert_eq!(calls(&out).len(), 1);
        let out = run(
            &request(Some("low"), json!("none")),
            vec![chunk(Some(NATIVE), Some("stop"), 0)],
        )
        .await;
        assert!(calls(&out).is_empty());
        assert_eq!(text(&out, "content"), NATIVE);
    }
}
