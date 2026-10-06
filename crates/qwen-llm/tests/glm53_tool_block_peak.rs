//! The buffered GLM-5.3 tool block's memory model
//! ([`TOOL_BLOCK_PEAK_FACTOR`]) against measured allocations: a counting
//! global allocator (this test binary only) records the high-water mark of
//! live bytes while a block is parsed and its calls are serialized as the
//! server publishes them. Container-heavy shapes (arrays of short scalars,
//! one-element arrays, deep nesting, many keys) are the worst cases: their
//! parsed trees cost far more than their text.

use qwen_llm::glm5_next_chat::{
    TOOL_BLOCK_PEAK_FACTOR, ToolDefinition, parse_tool_calls, tool_block_peak_bytes,
};
use serde_json::{Value, json};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Bytes an allocation really occupies: macOS malloc hands out at least
/// 16-byte quanta, so a one-digit number's text costs 16, not 1.
fn charged(size: usize) -> usize {
    size.max(1).next_multiple_of(16)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let size = charged(layout.size());
            let live = LIVE.fetch_add(size, Ordering::SeqCst) + size;
            PEAK.fetch_max(live, Ordering::SeqCst);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(charged(layout.size()), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Count a move as both blocks live (the worst case of a realloc).
        let size = charged(new_size);
        let grown = LIVE.fetch_add(size, Ordering::SeqCst) + size;
        PEAK.fetch_max(grown, Ordering::SeqCst);
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if moved.is_null() {
            LIVE.fetch_sub(size, Ordering::SeqCst);
        } else {
            LIVE.fetch_sub(charged(layout.size()), Ordering::SeqCst);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Peak live bytes above the block itself while the block is parsed and
/// every call is published the way serve does: arguments serialized once
/// (kept on the item), embedded in an event, and the event serialized.
fn publication_peak(block: &str, definitions: &[ToolDefinition]) -> usize {
    let baseline = LIVE.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    {
        let calls = parse_tool_calls(block, definitions).expect("block parses");
        let mut kept = Vec::new();
        for call in &calls {
            let arguments = serde_json::to_string(&call.arguments).unwrap();
            let event = json!({"type": "function_call", "name": call.name, "arguments": arguments});
            let line = serde_json::to_string(&event).unwrap();
            kept.push((arguments, event, line));
        }
        let response =
            json!({"output": kept.iter().map(|(_, event, _)| event.clone()).collect::<Vec<_>>()});
        let body = serde_json::to_string(&response).unwrap();
        std::hint::black_box((&calls, &kept, &response, &body));
    }
    PEAK.load(Ordering::SeqCst) - baseline
}

fn tool(parameters: Value) -> ToolDefinition {
    ToolDefinition::from_value(&json!({"name": "f", "parameters": parameters})).unwrap()
}

fn call(key: &str, value: &str) -> String {
    format!("<tool_call>f<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>")
}

fn repeat(item: &str, count: usize) -> String {
    vec![item; count].join(",")
}

#[test]
fn measured_publication_peak_stays_within_the_tool_block_model() {
    let array = tool(json!({"type": "object", "properties": {"a": {"type": "array"}}}));
    let object = tool(json!({"type": "object", "properties": {"a": {"type": "object"}}}));
    let string = tool(json!({"type": "object", "properties": {"a": {"type": "string"}}}));
    let open = tool(json!({"type": "object"}));
    let n = 20_000;
    let keys = (0..n)
        .map(|i| format!("\"k{i}\":1"))
        .collect::<Vec<_>>()
        .join(",");
    let many_args = (0..2_000)
        .map(|i| format!("<arg_key>a{i}</arg_key><arg_value>x</arg_value>"))
        .collect::<String>();
    // The decoder's nesting limit is 128 levels; the argument is one.
    let nested = format!("{}1{}", "[".repeat(127), "]".repeat(127));
    // One argument holding an array of arrays nested to the limit: the
    // densest container allocation per byte (a four-slot vector per `[]`).
    let inner = format!("{}1{}", "[".repeat(126), "]".repeat(126));
    let deep_array = format!("[{}]", repeat(&inner, 400));
    let nested_objects = format!("{}1{}", "{\"\":".repeat(127), "}".repeat(127));
    let cases: Vec<(&str, String, &ToolDefinition)> = vec![
        (
            "short numbers",
            call("a", &format!("[{}]", repeat("1", n))),
            &array,
        ),
        (
            "one-element arrays",
            call("a", &format!("[{}]", repeat("[1]", n))),
            &array,
        ),
        (
            "empty arrays",
            call("a", &format!("[{}]", repeat("[]", n))),
            &array,
        ),
        (
            "empty objects",
            call("a", &format!("[{}]", repeat("{}", n))),
            &array,
        ),
        (
            "one-key objects",
            call("a", &format!("[{}]", repeat("{\"a\":1}", n))),
            &array,
        ),
        (
            "short strings",
            call("a", &format!("[{}]", repeat("\"a\"", n))),
            &array,
        ),
        ("control escapes", call("a", &"\u{1}".repeat(n)), &string),
        ("deep nesting", call("a", &nested).repeat(200), &array),
        ("array of deep arrays", call("a", &deep_array), &array),
        (
            "deep object nesting",
            call("a", &nested_objects).repeat(200),
            &object,
        ),
        (
            "nested pairs",
            call("a", &format!("[{}]", repeat("[[1]]", n))),
            &array,
        ),
        ("many keys", call("a", &format!("{{{keys}}}")), &object),
        (
            "many arguments",
            format!("<tool_call>f{many_args}</tool_call>"),
            &open,
        ),
        ("many calls", call("a", "x").repeat(5_000), &open),
    ];
    let mut worst = 0.0f64;
    for (label, block, definition) in &cases {
        let peak = publication_peak(block, std::slice::from_ref(*definition));
        let ratio = peak as f64 / block.len() as f64;
        worst = worst.max(ratio);
        eprintln!(
            "[tool-block-peak] {label}: block={} peak={peak} ratio={ratio:.1}",
            block.len()
        );
        assert!(
            block.len() + peak <= tool_block_peak_bytes(block.len()),
            "{label}: {peak} bytes above a {}-byte block exceeds the model (factor {TOOL_BLOCK_PEAK_FACTOR})",
            block.len()
        );
    }
    eprintln!("[tool-block-peak] worst ratio {worst:.1} (model factor {TOOL_BLOCK_PEAK_FACTOR})");
}
