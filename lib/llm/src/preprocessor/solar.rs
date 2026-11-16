// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Solar Open2 tool-call streaming policy.
//!
//! Native (`auto`) calls stream as the model writes them, following the
//! reference streaming parser (UpstageAI/vllm `v0.22.0-solar-open2`
//! `solar_open2_tool_parser.py`): the name is sent when its line ends, each
//! argument when its value is known, and `}` at `<|tool_call:end|>`. String
//! values also stream while they are being written, so a long argument never
//! holds the stream. Prose after the first call is dropped, as in
//! `solar::parse`.
//!
//! Streamed deltas cannot be revoked, so executability is carried by the
//! terminal reason: `tool_calls` is reported only after a real backend `stop`
//! with at least one closed call. Length, repetition, cancellation, errors and
//! missing terminals keep their own reason, and the collected (non-streaming)
//! response drops every call in that case. A call the grammar rejects after its
//! name was sent is never closed, so its arguments never form valid JSON.
use super::*;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_parsers::tool_calling::solar::{ARG_END, ARG_START, ARG_VALUE, CALL_END, CALL_START};
use dynamo_protocols::types::{
    ChatCompletionMessageContent, ChatCompletionMessageToolCallChunk, FinishReason,
    FunctionCallStream, FunctionType,
};
use serde_json::Value;

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

/// The single commit rule shared by the streaming paths: a backend `stop`
/// with no earlier error, and at least one closed call.
fn terminal_reason(backend: FinishReason, failed: bool, closed_calls: bool) -> FinishReason {
    match backend {
        FinishReason::Stop if !failed && closed_calls => FinishReason::ToolCalls,
        // The jail can synthesize `tool_calls`; it is never a backend success.
        FinishReason::ToolCalls => FinishReason::Length,
        reason => reason,
    }
}

/// Collected responses: only a committed terminal exposes calls, and only the
/// calls whose arguments closed.
pub(crate) fn enforce_collected_tool_calls(
    finish_reason: &mut Option<FinishReason>,
    tool_calls: &mut Option<Vec<dynamo_protocols::types::ChatCompletionMessageToolCall>>,
) {
    if *finish_reason != Some(FinishReason::ToolCalls) {
        if tool_calls.take().is_some_and(|calls| !calls.is_empty()) {
            tracing::warn!(
                ?finish_reason,
                "dropping Solar Open2 tool calls without a successful backend stop"
            );
        }
        return;
    }
    if let Some(calls) = tool_calls.as_mut() {
        calls.retain(|call| {
            serde_json::from_str::<serde_json::Map<String, Value>>(&call.function.arguments).is_ok()
        });
    }
    if tool_calls.as_ref().is_none_or(Vec::is_empty) {
        *tool_calls = None;
        *finish_reason = Some(FinishReason::Stop);
    }
}

pub(super) fn apply_tool_policy<S>(
    stream: S,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    tools: Option<Vec<ToolDefinition>>,
    uses_structural_tag: bool,
    guided_streaming: bool,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    // Same format selection as the jail: a structural tag constrains the
    // native format; otherwise required/named are guided JSON.
    let named = match &tool_choice {
        Some(ChatCompletionToolChoiceOption::Named(named)) => Some(named.function.name.clone()),
        _ => None,
    };
    let guided_json = !uses_structural_tag
        && matches!(
            tool_choice,
            Some(
                ChatCompletionToolChoiceOption::Named(_) | ChatCompletionToolChoiceOption::Required
            )
        );
    if guided_json {
        Box::pin(guided_policy(
            stream,
            tool_choice,
            tools,
            uses_structural_tag,
            guided_streaming,
        ))
    } else {
        Box::pin(native_stream(stream, tools.unwrap_or_default(), named))
    }
}

/// Guided `required`/named JSON stays on the shared jail. Its deltas pass
/// through as released; only the terminal reason follows the commit rule.
fn guided_policy<S>(
    stream: S,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    tools: Option<Vec<ToolDefinition>>,
    uses_structural_tag: bool,
    guided_streaming: bool,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    #[derive(Default)]
    struct Backend {
        finishes: HashMap<u32, FinishReason>,
        failed: bool,
    }
    // The jail may synthesize a tool_calls finish at EOF. Remember only actual
    // backend terminal reasons, recorded before the jail. The lock is touched
    // only on terminal/error chunks.
    let backend = Arc::new(parking_lot::Mutex::new(Backend::default()));
    let input_backend = backend.clone();
    let stream = stream.map(move |response| {
        if response.error.is_some() || response.event.as_deref() == Some("error") {
            input_backend.lock().failed = true;
        }
        if let Some(data) = &response.data {
            for choice in &data.inner.choices {
                if let Some(reason) = choice.finish_reason {
                    input_backend.lock().finishes.insert(choice.index, reason);
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
        guided_streaming,
        stream,
    );
    let mut tool_section = HashSet::new();
    parsed.map(move |mut response| {
        if let Some(data) = response.data.as_mut() {
            for choice in &mut data.inner.choices {
                if tool_section.contains(&choice.index) {
                    choice.delta.content = None;
                }
                if choice
                    .delta
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
                {
                    tool_section.insert(choice.index);
                }
                if choice.finish_reason.is_some() {
                    let state = backend.lock();
                    // A missing backend terminal is an interrupted generation.
                    let reason = state
                        .finishes
                        .get(&choice.index)
                        .copied()
                        .unwrap_or(FinishReason::Length);
                    choice.finish_reason = Some(terminal_reason(
                        reason,
                        state.failed,
                        tool_section.contains(&choice.index),
                    ));
                }
            }
        }
        response
    })
}

fn native_stream<S>(
    stream: S,
    tools: Vec<ToolDefinition>,
    named: Option<String>,
) -> impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
{
    let tools: Arc<[ToolDefinition]> = tools.into();
    async_stream::stream! {
        let mut parsers: HashMap<u32, NativeParser> = HashMap::new();
        let mut finished: HashSet<u32> = HashSet::new();
        let mut failed = false;
        let mut template: Option<NvCreateChatCompletionStreamResponse> = None;
        tokio::pin!(stream);
        while let Some(mut response) = stream.next().await {
            if response.error.is_some() || response.event.as_deref() == Some("error") {
                failed = true;
            }
            let Some(data) = response.data.as_mut() else {
                yield response;
                continue;
            };
            if data.inner.choices.is_empty() {
                // OpenAI ordering: terminals precede the usage-only chunk.
                if let Some(template) = &template {
                    for terminal in finish_unterminated(&mut parsers, &mut finished, template) {
                        yield terminal;
                    }
                }
                yield response;
                continue;
            }
            if template.is_none() {
                let mut empty = data.clone();
                empty.inner.choices.clear();
                empty.inner.usage = None;
                template = Some(empty);
            }
            for choice in &mut data.inner.choices {
                let parser = parsers
                    .entry(choice.index)
                    .or_insert_with(|| NativeParser::new(tools.clone(), named.clone()));
                let mut out = Output::default();
                let text = match choice.delta.content.take() {
                    Some(ChatCompletionMessageContent::Text(text)) => {
                        parser.push(&text, &mut out);
                        true
                    }
                    other => {
                        choice.delta.content = other;
                        false
                    }
                };
                if let Some(reason) = choice.finish_reason
                    && finished.insert(choice.index)
                {
                    parser.finish(reason, &mut out);
                    choice.finish_reason =
                        Some(terminal_reason(reason, failed, parser.closed_calls > 0));
                }
                if text || !out.content.is_empty() {
                    choice.delta.content = (!out.content.is_empty())
                        .then_some(ChatCompletionMessageContent::Text(out.content));
                }
                if !out.calls.is_empty() {
                    choice.delta.tool_calls = Some(out.calls);
                }
            }
            yield response;
        }
        if let Some(template) = &template {
            for terminal in finish_unterminated(&mut parsers, &mut finished, template) {
                yield terminal;
            }
        }
    }
}

/// A choice without a backend terminal was interrupted. Close it as `length`
/// once it started a call; plain text keeps the upstream shape.
fn finish_unterminated(
    parsers: &mut HashMap<u32, NativeParser>,
    finished: &mut HashSet<u32>,
    template: &NvCreateChatCompletionStreamResponse,
) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    let mut indices: Vec<_> = parsers
        .keys()
        .copied()
        .filter(|index| !finished.contains(index))
        .collect();
    indices.sort_unstable();
    let mut out = Vec::new();
    for index in indices {
        finished.insert(index);
        let parser = parsers.get_mut(&index).expect("index from the parser map");
        let started = parser.next_index > 0 || parser.state != State::Prose;
        let mut emitted = Output::default();
        // Held plain text is still the model's text; only a started call
        // turns the missing terminal into an interruption.
        parser.finish(
            if started {
                FinishReason::Length
            } else {
                FinishReason::Stop
            },
            &mut emitted,
        );
        if !started && emitted.content.is_empty() {
            continue;
        }
        let content = (!emitted.content.is_empty())
            .then_some(ChatCompletionMessageContent::Text(emitted.content));
        let calls = (!emitted.calls.is_empty()).then_some(emitted.calls);
        out.push(
            crate::protocols::openai::chat_completions::stream_choice_chunk_from_template(
                template,
                index,
                content,
                None,
                calls,
                started.then_some(FinishReason::Length),
            ),
        );
    }
    out
}

#[derive(Default)]
struct Output {
    content: String,
    calls: Vec<ChatCompletionMessageToolCallChunk>,
}

impl Output {
    fn open(&mut self, index: u32, name: &str) {
        self.calls.push(ChatCompletionMessageToolCallChunk {
            custom: None,
            index,
            id: Some(format!("call-{}", uuid::Uuid::new_v4())),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some(name.to_string()),
                arguments: Some(String::new()),
            }),
        });
    }

    /// Append an argument fragment, folding into this chunk's last delta for
    /// the same call.
    fn args(&mut self, index: u32, fragment: &str) {
        if let Some(last) = self.calls.last_mut()
            && last.index == index
            && let Some(function) = last.function.as_mut()
        {
            function
                .arguments
                .get_or_insert_with(String::new)
                .push_str(fragment);
            return;
        }
        self.calls.push(ChatCompletionMessageToolCallChunk {
            custom: None,
            index,
            id: None,
            r#type: None,
            function: Some(FunctionCallStream {
                name: None,
                arguments: Some(fragment.to_string()),
            }),
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Before the first call (content) or between calls (dropped).
    Prose,
    /// After `<|tool_call:start|>`, reading the name line.
    Name,
    /// Inside a call, expecting an argument or the closing fence.
    Call,
    /// After `<|tool_arg:start|>`, reading the key.
    Key,
    /// After `<|tool_arg:value|>`, reading the value.
    Value,
    /// A call the grammar rejects (or a filtered name): skip to its fence.
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueMode {
    /// A string-typed value that could still be the literal `null`.
    Undecided,
    /// A string value whose opening quote was sent.
    Streaming,
    /// A typed value converted once it is complete.
    Whole,
}

struct NativeParser {
    tools: Arc<[ToolDefinition]>,
    named: Option<String>,
    buffer: String,
    state: State,
    /// Any `<|tool_call:start|>` seen: later prose is not content.
    calls_seen: bool,
    next_index: u32,
    closed_calls: usize,
    /// The emitted call being parsed (`None` while skipping).
    call: Option<u32>,
    name: String,
    first_arg: bool,
    key: String,
    mode: ValueMode,
    raw: String,
}

impl NativeParser {
    fn new(tools: Arc<[ToolDefinition]>, named: Option<String>) -> Self {
        Self {
            tools,
            named,
            buffer: String::new(),
            state: State::Prose,
            calls_seen: false,
            next_index: 0,
            closed_calls: 0,
            call: None,
            name: String::new(),
            first_arg: true,
            key: String::new(),
            mode: ValueMode::Whole,
            raw: String::new(),
        }
    }

    fn push(&mut self, text: &str, out: &mut Output) {
        self.buffer.push_str(text);
        loop {
            let progressed = match self.state {
                State::Prose => self.prose(out),
                State::Name => self.read_name(out),
                State::Call => self.in_call(out),
                State::Key => self.read_key(),
                State::Value => self.read_value(out),
                State::Skip => self.skip(),
            };
            if !progressed {
                break;
            }
        }
    }

    /// End of generation. Text held as a possible fence prefix is content only
    /// before the first call; an unclosed call stays unclosed.
    fn finish(&mut self, reason: FinishReason, out: &mut Output) {
        if self.state == State::Prose && !self.calls_seen && reason != FinishReason::Length {
            out.content.push_str(&self.buffer);
        }
        if self.state != State::Prose && self.call.is_some() {
            tracing::warn!(
                ?reason,
                name = %self.name,
                "Solar Open2 tool call ended before <|tool_call:end|>"
            );
        }
        self.buffer.clear();
        self.state = State::Prose;
        self.call = None;
    }

    fn prose(&mut self, out: &mut Output) -> bool {
        if let Some(at) = self.buffer.find(CALL_START) {
            if !self.calls_seen {
                out.content.push_str(&self.buffer[..at]);
            }
            self.buffer.drain(..at + CALL_START.len());
            self.calls_seen = true;
            self.state = State::Name;
            return true;
        }
        let keep = holdback(&self.buffer, &[CALL_START]);
        let flush = self.buffer.len() - keep;
        if !self.calls_seen {
            out.content.push_str(&self.buffer[..flush]);
        }
        self.buffer.drain(..flush);
        false
    }

    fn read_name(&mut self, out: &mut Output) -> bool {
        let newline = self.buffer.find('\n');
        let fence = self.buffer.find(CALL_END);
        match (newline, fence) {
            // No name line inside the call body: `solar::parse` skips it.
            (newline, Some(end)) if newline.is_none_or(|nl| end < nl) => {
                self.buffer.drain(..end + CALL_END.len());
                self.state = State::Prose;
                true
            }
            (Some(nl), _) => {
                let name = self.buffer[..nl].trim().to_string();
                self.buffer.drain(..nl + 1);
                let filtered = self.named.as_ref().is_some_and(|named| *named != name);
                if name.is_empty() || name.contains(CALL_START) || filtered {
                    self.call = None;
                    self.state = State::Skip;
                    return true;
                }
                let index = self.next_index;
                self.next_index += 1;
                out.open(index, &name);
                self.call = Some(index);
                self.name = name;
                self.first_arg = true;
                self.state = State::Call;
                true
            }
            _ => false,
        }
    }

    fn in_call(&mut self, out: &mut Output) -> bool {
        let rest = self.buffer.trim_start();
        if let Some(after) = rest.strip_prefix(ARG_START) {
            self.buffer = after.to_string();
            self.state = State::Key;
            return true;
        }
        if let Some(after) = rest.strip_prefix(CALL_END) {
            let index = self.call.take().expect("an emitted call is open");
            out.args(index, if self.first_arg { "{}" } else { "}" });
            self.closed_calls += 1;
            self.buffer = after.to_string();
            self.state = State::Prose;
            return true;
        }
        if rest.is_empty() || ARG_START.starts_with(rest) || CALL_END.starts_with(rest) {
            self.buffer = rest.to_string();
            return false;
        }
        self.reject("text between arguments");
        true
    }

    fn read_key(&mut self) -> bool {
        let value = self.buffer.find(ARG_VALUE);
        let fence = self.buffer.find(CALL_END);
        match (value, fence) {
            (value, Some(end)) if value.is_none_or(|v| end < v) => {
                self.reject_at_fence(end, "argument without a value");
                true
            }
            (Some(v), _) => {
                let key = self.buffer[..v].to_string();
                self.buffer.drain(..v + ARG_VALUE.len());
                if key.is_empty() || key.contains(ARG_START) {
                    self.reject("invalid argument name");
                    return true;
                }
                self.mode = if self.streams_as_string(&key) {
                    ValueMode::Undecided
                } else {
                    ValueMode::Whole
                };
                self.key = key;
                self.raw.clear();
                self.state = State::Value;
                true
            }
            _ => false,
        }
    }

    fn read_value(&mut self, out: &mut Output) -> bool {
        let index = self.call.expect("an emitted call is open");
        let arg_end = self.buffer.find(ARG_END);
        let fence = self.buffer.find(CALL_END);
        match (arg_end, fence) {
            (end, Some(fence)) if end.is_none_or(|e| fence < e) => {
                self.reject_at_fence(fence, "argument value without its end");
                true
            }
            (Some(end), _) => {
                let piece = self.buffer[..end].to_string();
                self.buffer.drain(..end + ARG_END.len());
                if self.mode == ValueMode::Streaming {
                    out.args(index, &escape(&piece));
                    out.args(index, "\"");
                } else {
                    self.raw.push_str(&piece);
                    let value = self.coerce();
                    let fragment = format!(
                        "{}{}:{}",
                        if self.first_arg { "{" } else { "," },
                        json_string(&self.key),
                        value
                    );
                    out.args(index, &fragment);
                }
                self.first_arg = false;
                self.state = State::Call;
                true
            }
            _ => {
                let keep = holdback(&self.buffer, &[ARG_END, CALL_END]);
                let available: String = self.buffer.drain(..self.buffer.len() - keep).collect();
                match self.mode {
                    ValueMode::Streaming => {
                        if !available.is_empty() {
                            out.args(index, &escape(&available));
                        }
                    }
                    ValueMode::Whole => self.raw.push_str(&available),
                    ValueMode::Undecided => {
                        self.raw.push_str(&available);
                        if !could_be_null(&self.raw) {
                            let fragment = format!(
                                "{}{}:\"{}",
                                if self.first_arg { "{" } else { "," },
                                json_string(&self.key),
                                escape(&self.raw)
                            );
                            out.args(index, &fragment);
                            self.first_arg = false;
                            self.raw.clear();
                            self.mode = ValueMode::Streaming;
                        }
                    }
                }
                false
            }
        }
    }

    fn skip(&mut self) -> bool {
        if let Some(end) = self.buffer.find(CALL_END) {
            self.buffer.drain(..end + CALL_END.len());
            self.state = State::Prose;
            return true;
        }
        let keep = holdback(&self.buffer, &[CALL_END]);
        self.buffer.drain(..self.buffer.len() - keep);
        false
    }

    /// The grammar rejects the open call; it is never closed.
    fn reject(&mut self, why: &str) {
        if self.call.take().is_some() {
            tracing::warn!(name = %self.name, why, "rejecting malformed Solar Open2 tool call");
        }
        self.state = State::Skip;
    }

    fn reject_at_fence(&mut self, fence: usize, why: &str) {
        self.reject(why);
        self.buffer.drain(..fence + CALL_END.len());
        self.state = State::Prose;
    }

    /// Mirrors `solar::coerce`'s type selection; only string-typed values can
    /// stream before they are complete.
    fn streams_as_string(&self, key: &str) -> bool {
        let ty = self
            .tools
            .iter()
            .find(|tool| tool.name == self.name)
            .and_then(|tool| tool.parameters.as_ref())
            .and_then(|schema| schema.get("properties"))
            .and_then(|props| props.get(key))
            .and_then(|spec| spec.get("type"));
        let ty = match ty {
            Some(Value::Array(types)) => types
                .iter()
                .filter_map(Value::as_str)
                .find(|ty| *ty != "null"),
            Some(ty) => ty.as_str(),
            None => None,
        }
        .unwrap_or("string");
        matches!(
            ty.trim().to_ascii_lowercase().as_str(),
            "string" | "str" | "text" | "varchar" | "char" | "enum"
        )
    }

    /// Convert one complete value with the grammar's own coercion.
    fn coerce(&self) -> String {
        let call = format!(
            "{CALL_START}{}\n{ARG_START}{}{ARG_VALUE}{}{ARG_END}\n{CALL_END}",
            self.name, self.key, self.raw
        );
        dynamo_parsers::tool_calling::solar::parse(&call, Some(&self.tools))
            .ok()
            .and_then(|(calls, _)| calls.into_iter().next())
            .and_then(|call| {
                serde_json::from_str::<serde_json::Map<String, Value>>(&call.function.arguments)
                    .ok()
            })
            .and_then(|args| args.into_iter().next())
            .map(|(_, value)| value)
            .unwrap_or_else(|| Value::String(self.raw.clone()))
            .to_string()
    }
}

/// Trailing bytes that may begin one of `fences`; they wait for the next chunk.
fn holdback(text: &str, fences: &[&str]) -> usize {
    let Some(start) = text.rfind('<') else {
        return 0;
    };
    let tail = &text[start..];
    if fences
        .iter()
        .any(|fence| tail.len() < fence.len() && fence.starts_with(tail))
    {
        tail.len()
    } else {
        0
    }
}

/// `solar::coerce` maps a trimmed, case-insensitive `null` to JSON null.
fn could_be_null(raw: &str) -> bool {
    let value = raw.trim();
    value.len() <= 4 && "null".starts_with(&value.to_ascii_lowercase())
}

fn json_string(text: &str) -> String {
    Value::String(text.to_string()).to_string()
}

/// JSON string escaping is per character, so escaped pieces concatenate to
/// the escaped whole.
fn escape(text: &str) -> String {
    let quoted = json_string(text);
    quoted[1..quoted.len() - 1].to_string()
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

    /// Streamed tool-call chunks folded per call index, as a client would.
    fn calls(out: &[Value]) -> Vec<Value> {
        merged_calls(out)
    }

    fn merged_calls(out: &[Value]) -> Vec<Value> {
        let mut calls: std::collections::BTreeMap<u64, Value> = Default::default();
        for chunk in out
            .iter()
            .flat_map(|v| v["choices"].as_array().into_iter().flatten())
            .flat_map(|c| c["delta"]["tool_calls"].as_array().into_iter().flatten())
        {
            let index = chunk["index"].as_u64().unwrap();
            let call = calls.entry(index).or_insert_with(|| {
                json!({"index": index, "id": null, "type": null,
                       "function": {"name": null, "arguments": ""}})
            });
            for key in ["id", "type"] {
                if !chunk[key].is_null() {
                    assert!(call[key].is_null(), "duplicate {key}: {out:?}");
                    call[key] = chunk[key].clone();
                }
            }
            if !chunk["function"]["name"].is_null() {
                call["function"]["name"] = chunk["function"]["name"].clone();
            }
            if let Some(args) = chunk["function"]["arguments"].as_str() {
                let joined = format!("{}{args}", call["function"]["arguments"].as_str().unwrap());
                call["function"]["arguments"] = json!(joined);
            }
        }
        calls.into_values().collect()
    }

    fn finishes(out: &[Value]) -> Vec<String> {
        out.iter()
            .flat_map(|v| v["choices"].as_array().into_iter().flatten())
            .filter_map(|c| c["finish_reason"].as_str().map(str::to_string))
            .collect()
    }

    async fn collect(
        req: &NvCreateChatCompletionRequest,
        chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
    ) -> Value {
        let parsed = preprocessor()
            .postprocessor_parsing_stream(stream::iter(chunks), req, false, false)
            .unwrap();
        let options = crate::protocols::openai::ParsingOptions::new(
            Some("solar_open2".into()),
            Some("solar_open2".into()),
        );
        let response = NvCreateChatCompletionResponse::from_annotated_stream(parsed, options)
            .await
            .unwrap();
        serde_json::to_value(response).unwrap()
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
    async fn solar_truncation_never_commits_calls() {
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
                let req = request(Some("low"), choice.clone());
                let input = vec![
                    chunk(Some(raw), None, 0),
                    chunk(Some("<|tool_call:start|>cut"), finish, 0),
                ];
                let out = run(&req, input.clone()).await;
                // Deltas may already be out; executability is the terminal.
                let terminals = finishes(&out);
                assert!(
                    !terminals.contains(&"tool_calls".to_string()),
                    "{choice} {finish:?}: {out:?}"
                );
                assert_eq!(
                    terminals.last().map(String::as_str),
                    Some(finish.unwrap_or("length")),
                    "{choice} {finish:?}: {out:?}"
                );
                let collected = collect(&req, input).await;
                assert!(
                    collected["choices"][0]["message"]["tool_calls"].is_null(),
                    "{choice} {finish:?}: {collected}"
                );
            }
        }
    }

    #[tokio::test]
    async fn solar_calls_stream_before_backend_stop_and_choice_isolation() {
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
            false,
        );
        tokio::pin!(output);
        tx.send(chunk(Some(NATIVE), None, 0)).await.unwrap();
        // The complete call is delivered without waiting for the backend.
        let first = serde_json::to_value(output.next().await.unwrap().data).unwrap();
        let streamed = merged_calls(std::slice::from_ref(&first));
        assert_eq!(streamed.len(), 1, "{first}");
        assert_eq!(
            serde_json::from_str::<Value>(streamed[0]["function"]["arguments"].as_str().unwrap())
                .unwrap(),
            json!({"query":"서울","limit":"7"})
        );
        assert!(first["choices"][0]["finish_reason"].is_null());
        tx.send(chunk(Some(NATIVE), Some("length"), 1))
            .await
            .unwrap();
        tx.send(chunk(None, Some("stop"), 0)).await.unwrap();
        drop(tx);
        let mut out = vec![first];
        out.extend(
            output
                .map(|out| serde_json::to_value(out.data).unwrap())
                .collect::<Vec<_>>()
                .await,
        );
        let finish_of = |index: u64| {
            out.iter()
                .flat_map(|v| v["choices"].as_array().into_iter().flatten())
                .filter(|c| c["index"] == index && !c["finish_reason"].is_null())
                .map(|c| c["finish_reason"].clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(finish_of(0), vec![json!("tool_calls")], "{out:?}");
        assert_eq!(finish_of(1), vec![json!("length")], "{out:?}");
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
        assert!(
            !finishes(&out).contains(&"tool_calls".to_string()),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn solar_backend_error_never_commits_calls() {
        let input = vec![
            chunk(Some(NATIVE), None, 0),
            Annotated::from_error("backend failed"),
            chunk(None, Some("stop"), 0),
        ];
        let out = run(&request(Some("low"), json!("auto")), input).await;
        assert!(
            !finishes(&out).contains(&"tool_calls".to_string()),
            "{out:?}"
        );
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
    async fn solar_parallel_calls_keep_ids_indices_and_arguments() {
        let raw = format!("{NATIVE}{NATIVE}");
        for per_char in [false, true] {
            let mut input = if per_char {
                raw.chars()
                    .map(|ch| chunk(Some(&ch.to_string()), None, 0))
                    .collect::<Vec<_>>()
            } else {
                vec![chunk(Some(&raw), None, 0)]
            };
            input.push(chunk(None, Some("stop"), 0));
            let out = run(&request(Some("low"), json!("auto")), input).await;
            let parsed = calls(&out);
            assert_eq!(parsed.len(), 2, "{out:?}");
            assert_ne!(parsed[0]["id"], parsed[1]["id"]);
            for (index, call) in parsed.iter().enumerate() {
                assert_eq!(call["index"], index);
                assert_eq!(call["function"]["name"], "search");
                assert_eq!(
                    serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap())
                        .unwrap(),
                    json!({"query":"서울","limit":7})
                );
            }
            assert!(text(&out, "content").is_empty());
        }
    }

    fn small_chunks(raw: &str) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
        let chars: Vec<char> = raw.chars().collect();
        let pieces: Vec<String> = chars.chunks(3).map(|c| c.iter().collect()).collect();
        let last = pieces.len() - 1;
        pieces
            .iter()
            .enumerate()
            .map(|(i, piece)| chunk(Some(piece), (i == last).then_some("stop"), 0))
            .collect()
    }

    /// Customer report: XML-style text while thinking ended the thought and
    /// leaked the rest of it, and the answer, into content.
    #[tokio::test]
    async fn solar_literal_tool_call_text_in_thought_stays_reasoning() {
        let thought =
            "I could reply with <tool_call>{\"name\": \"x\"}</tool_call>, but no. More thinking.";
        for choice in [json!("auto"), json!("none")] {
            let out = run(
                &request(None, choice.clone()),
                small_chunks(&format!("{thought}<|think:end|>Final answer.")),
            )
            .await;
            assert_eq!(text(&out, "reasoning_content"), thought, "{choice}");
            assert_eq!(text(&out, "content"), "Final answer.", "{choice}");
            assert!(calls(&out).is_empty(), "{choice}");
        }
    }

    /// A complete call written inside the thought runs, after the real answer.
    #[tokio::test]
    async fn solar_call_inside_the_thought_runs_after_the_answer() {
        let out = run(
            &request(None, json!("auto")),
            small_chunks(&format!("plan\n{NATIVE}\n<|think:end|>확인할게요.")),
        )
        .await;
        assert_eq!(text(&out, "reasoning_content"), "plan");
        assert_eq!(text(&out, "content").trim_end(), "확인할게요.");
        let calls = calls(&out);
        assert_eq!(calls.len(), 1, "{out:?}");
        assert_eq!(calls[0]["function"]["name"], "search");
        assert!(
            out.iter()
                .flat_map(|v| v["choices"].as_array().into_iter().flatten())
                .any(|c| c["finish_reason"] == "tool_calls"),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn solar_tool_escape_and_tool_choice_none() {
        let out = run(
            &request(None, json!("auto")),
            vec![chunk(Some(&format!("plan {NATIVE}")), Some("stop"), 0)],
        )
        .await;
        // The call written in the thought moves to content (reference: the
        // remaining thought is stripped).
        assert_eq!(text(&out, "reasoning_content"), "plan");
        assert_eq!(calls(&out).len(), 1);
        let out = run(
            &request(Some("low"), json!("none")),
            vec![chunk(Some(NATIVE), Some("stop"), 0)],
        )
        .await;
        assert!(calls(&out).is_empty());
        assert_eq!(text(&out, "content"), NATIVE);
    }

    fn typed_request() -> NvCreateChatCompletionRequest {
        let body = json!({"model":"solar", "messages":[{"role":"user","content":"hello"}],
        "reasoning_effort":"low", "tool_choice":"auto",
        "tools":[
            {"type":"function","function":{"name":"search","parameters":{"type":"object",
                "properties":{"query":{"type":"string"},"limit":{"type":"integer"}}}}},
            {"type":"function","function":{"name":"write_file","parameters":{"type":"object",
                "properties":{"path":{"type":"string"},"content":{"type":"string"}}}}},
            {"type":"function","function":{"name":"cfg","parameters":{"type":"object",
                "properties":{"flag":{"type":"boolean"},"data":{"type":"object"},
                    "ratio":{"type":"number"},"opt":{"type":["null","integer"]},
                    "label":{"type":["string","null"]}}}}}
        ]});
        let mut request: NvCreateChatCompletionRequest = serde_json::from_value(body).unwrap();
        request.normalize_reasoning_template_args().unwrap();
        request
    }

    fn call_text(name: &str, args: &[(&str, &str)]) -> String {
        let mut text = format!("<|tool_call:start|>{name}\n");
        for (key, value) in args {
            text.push_str(&format!(
                "<|tool_arg:start|>{key}<|tool_arg:value|>{value}<|tool_arg:end|>\n"
            ));
        }
        text.push_str("<|tool_call:end|>");
        text
    }

    /// Outputs on which streamed calls must equal `solar::parse`.
    fn parse_corpus() -> Vec<String> {
        let file = "import os\n\ndef main():\n    print(\"a\\tb\", '<' , \"<|x|>\")\n    \
                    return {\"k\": [1, 2]}  # 서울 ✓\n";
        vec![
            NATIVE.to_string(),
            format!("요청을 처리합니다.\n{NATIVE}{NATIVE}"),
            call_text("write_file", &[("path", "main.py"), ("content", file)]),
            call_text(
                "write_file",
                &[(
                    "content",
                    "a <|tool_arg:start|> b <|tool_arg:value|> c <|tool_call:start|>",
                )],
            ),
            call_text("search", &[("query", "null"), ("limit", " 8 ")]),
            call_text("search", &[("query", " NULL "), ("limit", "many")]),
            call_text("search", &[("query", "nullable")]),
            call_text("search", &[("query", "nul")]),
            call_text("search", &[("query", "")]),
            call_text("search", &[("query", "   ")]),
            call_text(
                "cfg",
                &[
                    ("flag", "yes"),
                    ("data", "{\"a\": [1, 2], \"b\": null}"),
                    ("ratio", "2.0"),
                    ("opt", "null"),
                    ("label", "null"),
                ],
            ),
            call_text("cfg", &[("ratio", "1.25"), ("opt", "5"), ("label", "x")]),
            call_text("ping", &[]),
            call_text("unknown_tool", &[("x", "1"), ("y", "true")]),
            format!(
                "{}\n  {}",
                call_text("search", &[("query", "a")]),
                call_text("ping", &[])
            ),
            "<|tool_call:start|>search\n  <|tool_arg:start|>query<|tool_arg:value|>q\
             <|tool_arg:end|>   \n\n<|tool_call:end|>"
                .to_string(),
            format!("{NATIVE} trailing prose is not content"),
        ]
    }

    async fn assert_matches_parse(
        raw: &str,
        input: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
        label: &str,
    ) {
        let req = typed_request();
        let tools = crate::preprocessor::tool_choice::effective_tool_definitions(&req).unwrap();
        let (expected, content) =
            dynamo_parsers::tool_calling::solar::parse(raw, Some(&tools)).unwrap();
        let out = run(&req, input).await;
        let streamed = calls(&out);
        assert_eq!(streamed.len(), expected.len(), "{label}: {out:?}");
        for (got, want) in streamed.iter().zip(&expected) {
            assert_eq!(got["function"]["name"], want.function.name, "{label}");
            assert_eq!(
                got["function"]["arguments"].as_str().unwrap(),
                want.function.arguments,
                "{label}"
            );
        }
        assert_eq!(
            text(&out, "content"),
            content.unwrap_or_default(),
            "{label}: {out:?}"
        );
        let terminal = finishes(&out);
        assert_eq!(
            terminal,
            vec![
                if expected.is_empty() {
                    "stop"
                } else {
                    "tool_calls"
                }
                .to_string()
            ],
            "{label}: {out:?}"
        );
    }

    /// Every chunk boundary must produce what the whole-text grammar produces.
    #[tokio::test]
    async fn solar_streamed_calls_match_parse_at_every_split() {
        for raw in parse_corpus() {
            assert_matches_parse(&raw, vec![chunk(Some(&raw), Some("stop"), 0)], "whole").await;
            let per_char: Vec<_> = raw
                .chars()
                .map(|ch| chunk(Some(&ch.to_string()), None, 0))
                .chain(std::iter::once(chunk(None, Some("stop"), 0)))
                .collect();
            assert_matches_parse(&raw, per_char, "per char").await;
            for (split, _) in raw.char_indices().skip(1) {
                let input = vec![
                    chunk(Some(&raw[..split]), None, 0),
                    chunk(Some(&raw[split..]), Some("stop"), 0),
                ];
                assert_matches_parse(&raw, input, &format!("split {split} of {raw:?}")).await;
            }
        }
    }

    /// A long string argument streams while it is written.
    #[tokio::test]
    async fn solar_long_string_argument_streams_incrementally() {
        let content: String = (0..200).map(|i| format!("    value_{i} = {i}\n")).collect();
        let raw = call_text("write_file", &[("path", "big.py"), ("content", &content)]);
        let chars: Vec<char> = raw.chars().collect();
        let mut input: Vec<_> = chars
            .chunks(5)
            .map(|piece| chunk(Some(&piece.iter().collect::<String>()), None, 0))
            .collect();
        input.push(chunk(None, Some("stop"), 0));
        let total = input.len();
        let out = run(&typed_request(), input).await;
        assert_eq!(out.len(), total, "one output chunk per input chunk");
        let with_args: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, v)| {
                v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .is_some_and(|args| !args.is_empty())
            })
            .map(|(i, _)| i)
            .collect();
        // Name at the first newline, then arguments throughout, not at the end.
        assert!(
            with_args.len() > total / 2,
            "{} of {total}",
            with_args.len()
        );
        assert!(with_args[0] < 15, "first argument chunk {}", with_args[0]);
        let streamed = calls(&out);
        assert_eq!(streamed.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(streamed[0]["function"]["arguments"].as_str().unwrap())
                .unwrap(),
            json!({"path":"big.py","content":content})
        );
        assert_eq!(finishes(&out), vec!["tool_calls"]);
    }

    /// A call the grammar rejects after its name was sent is never closed and
    /// never executable; valid neighbours still commit.
    #[tokio::test]
    async fn solar_rejected_call_is_never_closed() {
        let broken = "<|tool_call:start|>search\njunk<|tool_arg:start|>query<|tool_arg:value|>x\
                      <|tool_arg:end|>\n<|tool_call:end|>";
        let unterminated_value = "<|tool_call:start|>search\n<|tool_arg:start|>query<|tool_arg:value|>x<|tool_call:end|>";
        let req = typed_request();
        for (raw, valid, terminal) in [
            (format!("{broken}{NATIVE}"), 1, "tool_calls"),
            (format!("{unterminated_value}{NATIVE}"), 1, "tool_calls"),
            (broken.to_string(), 0, "stop"),
            (
                "<|tool_call:start|><|tool_call:end|>".to_string(),
                0,
                "stop",
            ),
        ] {
            let input = small_chunks(&raw);
            let out = run(&req, input.clone()).await;
            assert_eq!(finishes(&out), vec![terminal], "{raw}: {out:?}");
            let closed = calls(&out)
                .iter()
                .filter(|call| {
                    serde_json::from_str::<serde_json::Map<String, Value>>(
                        call["function"]["arguments"].as_str().unwrap(),
                    )
                    .is_ok()
                })
                .count();
            assert_eq!(closed, valid, "{raw}: {out:?}");
            let collected = collect(&req, input).await;
            let choice = &collected["choices"][0];
            assert_eq!(choice["finish_reason"], terminal, "{collected}");
            assert_eq!(
                choice["message"]["tool_calls"]
                    .as_array()
                    .map_or(0, Vec::len),
                valid,
                "{collected}"
            );
        }
    }

    /// Structural-tag output is native even for a named choice; other names
    /// are filtered before their first delta.
    #[tokio::test]
    async fn solar_structural_tag_named_choice_filters_native_calls() {
        let other = call_text("write_file", &[("path", "x")]);
        let input = vec![chunk(Some(&format!("{other}{NATIVE}")), Some("stop"), 0)];
        let named: ChatCompletionToolChoiceOption =
            serde_json::from_value(json!({"type":"function","function":{"name":"search"}}))
                .unwrap();
        let out: Vec<Value> =
            apply_tool_policy(stream::iter(input), Some(named), None, true, false)
                .map(|out| serde_json::to_value(out.data).unwrap())
                .collect()
                .await;
        let streamed = calls(&out);
        assert_eq!(streamed.len(), 1, "{out:?}");
        assert_eq!(streamed[0]["index"], 0);
        assert_eq!(streamed[0]["function"]["name"], "search");
        assert_eq!(finishes(&out), vec!["tool_calls"]);
    }

    /// A stream without any backend terminal is interrupted once a call began.
    #[tokio::test]
    async fn solar_missing_terminal_closes_as_length_before_usage() {
        let mut usage = chunk(None, None, 0);
        let data = usage.data.as_mut().unwrap();
        data.inner.choices.clear();
        data.inner.usage = Some(
            serde_json::from_value(
                json!({"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}),
            )
            .unwrap(),
        );
        let out = run(
            &typed_request(),
            vec![chunk(Some(&NATIVE[..60]), None, 0), usage],
        )
        .await;
        assert_eq!(finishes(&out), vec!["length"], "{out:?}");
        let last = out.last().unwrap();
        assert!(last["choices"].as_array().unwrap().is_empty(), "{out:?}");
    }

    /// End to end through the token decoder: a repetition loop inside a tool
    /// argument streams, is stopped by the frontend detector, cancels the
    /// engine, and never becomes executable.
    mod pipeline {
        use super::*;
        use crate::backend::Backend;
        use crate::backend::repetition::{RepetitionConfig, RepetitionRule};
        use crate::protocols::common::llm_backend::LLMEngineOutput;
        use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider};
        use dynamo_runtime::pipeline::{
            Error, ManyOut, Operator, ResponseStream, ServerStreamingEngine, SingleIn,
        };

        struct TokenEngine {
            token_ids: Vec<u32>,
            finish: Option<crate::protocols::common::FinishReason>,
            context: Arc<parking_lot::Mutex<Option<Arc<dyn AsyncEngineContext>>>>,
        }

        #[async_trait::async_trait]
        impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
            for TokenEngine
        {
            async fn generate(
                &self,
                request: SingleIn<PreprocessedRequest>,
            ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
                let context = request.context();
                *self.context.lock() = Some(context.clone());
                let last = self.token_ids.len() - 1;
                let finish = self.finish.clone();
                let outputs: Vec<_> = self
                    .token_ids
                    .iter()
                    .enumerate()
                    .map(|(i, id)| {
                        Annotated::from_data(LLMEngineOutput {
                            token_ids: vec![*id],
                            finish_reason: if i == last { finish.clone() } else { None },
                            ..Default::default()
                        })
                    })
                    .collect();
                Ok(ResponseStream::new(
                    Box::pin(stream::iter(outputs)),
                    context,
                ))
            }
        }

        struct DecoderLink {
            backend: Arc<Backend>,
            engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        }

        #[async_trait::async_trait]
        impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, Error>
            for DecoderLink
        {
            async fn generate(
                &self,
                request: SingleIn<PreprocessedRequest>,
            ) -> Result<ManyOut<Annotated<BackendOutput>>, Error> {
                Operator::generate(self.backend.as_ref(), request, self.engine.clone()).await
            }
        }

        struct Run {
            out: Vec<Value>,
            engine_stopped: bool,
        }

        async fn serve(
            output: &str,
            finish: Option<crate::protocols::common::FinishReason>,
            repetition: Option<Arc<RepetitionConfig>>,
        ) -> Run {
            let preprocessor = preprocessor();
            // The engine side needs a real vocabulary; the mock card's is tiny.
            let hf = tokenizers::Tokenizer::from_file(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
            ))
            .unwrap();
            let token_ids = hf.encode(output, false).unwrap().get_ids().to_vec();
            assert_eq!(
                hf.decode(&token_ids, false).unwrap(),
                output,
                "tokenizer round trip"
            );
            let tokenizer: Arc<dyn crate::tokenizers::traits::Tokenizer> =
                Arc::new(crate::tokenizers::HuggingFaceTokenizer::from_tokenizer(hf));
            let tokenizer = crate::tokenizers::Tokenizer::from(tokenizer);
            let context = Arc::new(parking_lot::Mutex::new(None));
            let engine = Arc::new(TokenEngine {
                token_ids,
                finish,
                context: context.clone(),
            });
            let link = Arc::new(DecoderLink {
                backend: Backend::from_tokenizer_with_repetition(tokenizer, repetition),
                engine,
            });
            let mut request = typed_request();
            request.inner.stream = Some(true);
            let stream = Operator::generate(preprocessor.as_ref(), SingleIn::new(request), link)
                .await
                .unwrap();
            let out = stream
                .filter_map(|item| async move { item.data })
                .map(|data| serde_json::to_value(data).unwrap())
                .collect::<Vec<_>>()
                .await;
            let engine_stopped = context
                .lock()
                .as_ref()
                .is_some_and(|context| context.is_stopped());
            Run {
                out,
                engine_stopped,
            }
        }

        fn rule() -> Arc<RepetitionConfig> {
            Arc::new(
                RepetitionConfig::new(vec![RepetitionRule {
                    min_pattern_size: 4,
                    max_pattern_size: 64,
                    min_count: 8,
                }])
                .unwrap(),
            )
        }

        fn arg_chunks(out: &[Value]) -> usize {
            out.iter()
                .filter(|v| {
                    v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                        .as_str()
                        .is_some_and(|args| !args.is_empty())
                })
                .count()
        }

        const HEAD: &str = "<|tool_call:start|>write_file\n<|tool_arg:start|>path<|tool_arg:value|>\
                            main.py<|tool_arg:end|>\n<|tool_arg:start|>content<|tool_arg:value|>\
                            import os\n\ndef main():\n";

        #[tokio::test]
        async fn repetition_inside_a_tool_argument_streams_then_stops() {
            let looped: String = std::iter::repeat_n("    x = compute(x)\n", 400).collect();
            let output = format!("{HEAD}{looped}");
            let run = serve(&output, None, Some(rule())).await;
            let streamed = calls(&run.out);
            assert_eq!(streamed.len(), 1, "{:?}", run.out);
            assert_eq!(streamed[0]["function"]["name"], "write_file");
            let args = streamed[0]["function"]["arguments"].as_str().unwrap();
            assert!(
                args.starts_with(r#"{"path":"main.py","content":"import os\n\ndef main():\n"#),
                "{args}"
            );
            // The loop was visible while it ran, then cut early.
            assert!(arg_chunks(&run.out) > 20, "{}", arg_chunks(&run.out));
            assert!(
                args.len() < looped.len() / 4,
                "stopped late: {}",
                args.len()
            );
            assert_eq!(
                finishes(&run.out),
                vec!["repetition"],
                "{:?}",
                run.out.last()
            );
            assert!(
                run.engine_stopped,
                "the repetition stop must cancel the engine"
            );
            assert!(
                serde_json::from_str::<Value>(args).is_err(),
                "never valid JSON"
            );
        }

        #[tokio::test]
        async fn repetition_finish_is_not_executable_when_collected() {
            let looped: String = std::iter::repeat_n("    x = compute(x)\n", 400).collect();
            let output = format!("{HEAD}{looped}");
            let run = serve(&output, None, Some(rule())).await;
            let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = run
                .out
                .into_iter()
                .map(|v| Annotated::from_data(serde_json::from_value(v).unwrap()))
                .collect();
            let options = crate::protocols::openai::ParsingOptions::new(
                Some("solar_open2".into()),
                Some("solar_open2".into()),
            );
            let response = NvCreateChatCompletionResponse::from_annotated_stream(
                stream::iter(chunks),
                options,
            )
            .await
            .unwrap();
            let value = serde_json::to_value(response).unwrap();
            assert_eq!(
                value["choices"][0]["finish_reason"], "repetition",
                "{value}"
            );
            assert!(
                value["choices"][0]["message"]["tool_calls"].is_null(),
                "{value}"
            );
        }

        #[tokio::test]
        async fn long_file_without_repetition_streams_and_commits() {
            let body: String = (0..300)
                .map(|i| format!("    v{i} = {i} * {i}\n"))
                .collect();
            let output = format!("{HEAD}{body}<|tool_arg:end|>\n<|tool_call:end|>");
            let run = serve(
                &output,
                Some(crate::protocols::common::FinishReason::Stop),
                Some(rule()),
            )
            .await;
            assert_eq!(
                finishes(&run.out),
                vec!["tool_calls"],
                "{:?}",
                run.out.last()
            );
            assert!(arg_chunks(&run.out) > 100, "{}", arg_chunks(&run.out));
            let streamed = calls(&run.out);
            assert_eq!(streamed.len(), 1);
            assert_eq!(
                serde_json::from_str::<Value>(
                    streamed[0]["function"]["arguments"].as_str().unwrap()
                )
                .unwrap(),
                json!({"path":"main.py","content":format!("import os\n\ndef main():\n{body}")})
            );
        }

        #[tokio::test]
        async fn loop_without_detection_runs_to_the_engine_limit() {
            let looped: String = std::iter::repeat_n("    x = compute(x)\n", 100).collect();
            let output = format!("{HEAD}{looped}");
            let run = serve(
                &output,
                Some(crate::protocols::common::FinishReason::Length),
                None,
            )
            .await;
            assert_eq!(finishes(&run.out), vec!["length"], "{:?}", run.out.last());
            let args = calls(&run.out)[0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .to_string();
            assert!(
                args.len() > looped.len(),
                "the whole loop streamed: {}",
                args.len()
            );
        }
    }
}
