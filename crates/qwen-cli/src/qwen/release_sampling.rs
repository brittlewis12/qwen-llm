//! Day-to-day sampling defaults for `qwen run` and `qwen serve`, decided by
//! release identity (`release_identity` owns detection). Each entry follows
//! the release's own published `generation_config.json` or model card; the
//! sampler has no penalties, so penalty-dependent card presets are not
//! represented. A request that chooses no seed gets a fresh one
//! ([`fresh_seed`]), reported so the draw can be reproduced; no seed is
//! hard-coded. Bench, lens, JSONL batches and sampling attribution keep their
//! own explicit contracts and do not consult this module.

use crate::release_identity::{QwenShape, QwenVersion, ReleaseIdentity};
use crate::{Args, ExplicitCliOptions};
use qwen_llm::sampling::SamplingConfig;

pub(crate) fn release_sampling(identity: ReleaseIdentity, seed: u64) -> SamplingConfig {
    match identity {
        // Qwen3.5-27B and Qwen3.5-122B-A10B generation_config.json.
        ReleaseIdentity::Qwen {
            version: QwenVersion::Qwen35,
            shape: QwenShape::Dense27b | QwenShape::Moe122bA10b,
        } => SamplingConfig::qwen3_release(0.6, seed),
        // Every other published Qwen3.x config (3.5-35B-A3B, 3.6, 3.8)
        // declares 1.0 / 20 / 0.95. Releases that publish none (Qwen3.5
        // 0.8B-9B) and unidentified Qwen models take the same convention: a
        // 2026-10-07 A/B on Qwen3.5-2B/9B against the 0.7/200/1/.05 chat
        // preset found it equal or better and loop-free.
        ReleaseIdentity::Qwen { .. } => SamplingConfig::qwen3_release(1.0, seed),
        ReleaseIdentity::FlashNext => SamplingConfig::qwen38_flash_next(seed),
        ReleaseIdentity::DeepSeekV4 => SamplingConfig::deepseek_v4_0731(seed),
        ReleaseIdentity::MuseGlimmer => SamplingConfig::muse_glimmer(seed),
        ReleaseIdentity::K2Horizon => SamplingConfig::k2_horizon(seed),
        ReleaseIdentity::Glm53Flash => SamplingConfig::glm5_next(seed),
    }
}

/// A fresh seed for a request that chose none: the standard library's
/// OS-seeded hasher keys mixed with a per-process counter and the clock.
/// Callers record it (stderr for `qwen run`, `x_qwen.stats` for serve).
pub(crate) fn fresh_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static DRAWS: AtomicU64 = AtomicU64::new(0);
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(DRAWS.fetch_add(1, Ordering::Relaxed));
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos()),
    );
    hasher.finish()
}

/// A sampling value as its shortest decimal, so 0.95f32 reports 0.95 rather
/// than 0.949999988 in JSON echoes and projections.
pub(crate) fn decimal(value: f32) -> f64 {
    value.to_string().parse().unwrap_or(f64::from(value))
}

/// The release decision as `qwen info --json` reports it.
pub(crate) fn projection(identity: ReleaseIdentity) -> serde_json::Value {
    let release = release_sampling(identity, 42);
    serde_json::json!({
        "identity": identity.label(),
        "temperature": decimal(release.temperature),
        "top_k": release.top_k,
        "top_p": decimal(release.top_p),
        "min_p": decimal(release.min_p),
    })
}

/// Fill each CLI sampling field the user did not pass from the release
/// preset, and draw a fresh seed when none was passed (returned, for the
/// caller to report). `--prompt-lookup` is a greedy-only accelerator, so an
/// omitted temperature means greedy there; an explicit positive one is a
/// conflict the decode-policy check reports.
pub(crate) fn apply_run_defaults(
    args: &mut Args,
    explicit: ExplicitCliOptions,
    release: SamplingConfig,
) -> Option<u64> {
    if !explicit.temperature {
        args.temperature = if args.prompt_lookup {
            0.0
        } else {
            release.temperature
        };
    }
    if !explicit.top_k {
        args.top_k = release.top_k;
    }
    if !explicit.top_p {
        args.top_p = release.top_p;
    }
    if !explicit.min_p {
        args.min_p = release.min_p;
    }
    (!explicit.seed).then(|| {
        args.seed = fresh_seed();
        args.seed
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::release_identity::QwenShape::*;
    use crate::release_identity::QwenVersion::*;
    use clap::{CommandFactory, FromArgMatches};
    use qwen_llm::gguf::GgufFile;
    use qwen_llm::model_family::ModelFamily;

    fn qwen(version: QwenVersion, shape: QwenShape) -> ReleaseIdentity {
        ReleaseIdentity::Qwen { version, shape }
    }

    #[test]
    fn decisions_follow_each_release_and_never_default_to_greedy() {
        let cool = SamplingConfig::qwen3_release(0.6, 42);
        let standard = SamplingConfig::qwen3_release(1.0, 42);
        assert_eq!(release_sampling(qwen(Qwen35, Dense27b), 42), cool);
        assert_eq!(release_sampling(qwen(Qwen35, Moe122bA10b), 42), cool);
        for identity in [
            qwen(Qwen35, Moe35bA3b),
            qwen(Qwen35, Other),
            qwen(Qwen36, Dense27b),
            qwen(Qwen36, Moe35bA3b),
            qwen(Qwen38, Dense27b),
            qwen(Qwen38, Moe122bA10b),
            qwen(Unknown, Dense27b),
            qwen(Unknown, Other),
        ] {
            assert_eq!(release_sampling(identity, 42), standard, "{identity:?}");
        }
        for identity in [
            qwen(Qwen35, Dense27b),
            qwen(Unknown, Other),
            ReleaseIdentity::FlashNext,
            ReleaseIdentity::DeepSeekV4,
            ReleaseIdentity::MuseGlimmer,
            ReleaseIdentity::K2Horizon,
            ReleaseIdentity::Glm53Flash,
        ] {
            let config = release_sampling(identity, 7);
            assert!(config.temperature > 0.0, "{identity:?}");
            assert_eq!(config.seed, 7);
            config.validate().unwrap();
        }
    }

    fn run_args(extra: &[&str]) -> (Args, ExplicitCliOptions) {
        let mut argv = vec!["qwen", "run", "-m", "model.gguf", "--user", "hello"];
        argv.extend_from_slice(extra);
        let matches = Args::command().try_get_matches_from(argv).unwrap();
        let (_, run_matches) = matches.subcommand().unwrap();
        let explicit = ExplicitCliOptions::from_matches(run_matches);
        let mut args = Args::from_arg_matches(&matches).unwrap();
        let invocation = crate::cli::normalize(&mut args);
        invocation.apply_option_overrides(&mut args);
        (args, explicit)
    }

    /// The legacy flat form, which owns `--prompt-lookup`.
    fn flat_args(extra: &[&str]) -> (Args, ExplicitCliOptions) {
        let mut argv = vec!["qwen", "-m", "model.gguf", "--prompt", "hello"];
        argv.extend_from_slice(extra);
        let matches = Args::command().try_get_matches_from(argv).unwrap();
        let explicit = ExplicitCliOptions::from_matches(&matches);
        (Args::from_arg_matches(&matches).unwrap(), explicit)
    }

    fn effective(args: &Args) -> (f32, usize, f32, f32) {
        (args.temperature, args.top_k, args.top_p, args.min_p)
    }

    #[test]
    fn run_defaults_fill_only_omitted_fields_and_prompt_lookup_implies_greedy() {
        let release = SamplingConfig::qwen3_release(0.6, 42);
        let (mut args, explicit) = run_args(&[]);
        let drawn = apply_run_defaults(&mut args, explicit, release);
        assert_eq!(effective(&args), (0.6, 20, 0.95, 0.0));
        // No seed is hard-coded: an omitted one is drawn fresh and reported.
        assert_eq!(drawn, Some(args.seed));
        let (mut other, explicit) = run_args(&[]);
        assert_ne!(apply_run_defaults(&mut other, explicit, release), drawn);

        let (mut args, explicit) = run_args(&["--temp", "0", "--min-p", "0.05", "--seed", "7"]);
        assert_eq!(apply_run_defaults(&mut args, explicit, release), None);
        assert_eq!(effective(&args), (0.0, 20, 0.95, 0.05));
        assert_eq!(args.seed, 7, "an explicit seed is kept");

        let (mut args, explicit) = flat_args(&[]);
        apply_run_defaults(&mut args, explicit, release);
        assert_eq!(effective(&args), (0.6, 20, 0.95, 0.0), "legacy form too");

        let (mut args, explicit) = flat_args(&["--prompt-lookup"]);
        apply_run_defaults(&mut args, explicit, release);
        assert_eq!(
            args.temperature, 0.0,
            "omitted temperature is greedy for prompt lookup"
        );
        crate::validate_sampling_decode_policy(
            crate::cli_sampling_config(&args).unwrap(),
            args.prompt_lookup,
        )
        .unwrap();

        let (mut args, explicit) = flat_args(&["--prompt-lookup", "--temp", "0.7"]);
        apply_run_defaults(&mut args, explicit, release);
        assert_eq!(
            args.temperature, 0.7,
            "an explicit conflict is kept for the policy check"
        );
        assert!(
            crate::validate_sampling_decode_policy(
                crate::cli_sampling_config(&args).unwrap(),
                args.prompt_lookup
            )
            .is_err()
        );
    }

    /// The decisions are keyed by identity, not read from converter metadata,
    /// but where a local release GGUF restates its generation_config in
    /// `general.sampling.*`, the two must agree (absent keys mean "off").
    #[test]
    fn decisions_agree_with_release_ggufs_that_restate_their_generation_config() {
        use qwen_llm::test_fixtures as fx;
        let mut paths: Vec<std::path::PathBuf> = [
            fx::A3B_Q4_K_M,
            fx::QWEN36_27B_Q4_K_M,
            fx::A3B_IQ4_XS,
            fx::A10B_Q4_K_XL,
            fx::DEEPSEEK_V4_IQ3_XXS,
            fx::QWEN4EXP_Q3_K_XL,
            fx::GLM53_FLASH_UD_IQ3_XXS,
        ]
        .iter()
        .filter_map(|fixture| fixture.path_or_skip())
        .collect();
        paths.extend(
            [
                "/Users/tito/models/Qwen3.5-27B-Q4_K_M.gguf",
                "/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf",
                "/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
            ]
            .map(std::path::PathBuf::from)
            .into_iter()
            .filter(|path| path.exists()),
        );
        let mut checked = 0;
        for path in paths {
            let gguf = GgufFile::open(&path).unwrap();
            let Some(declared_temp) = gguf.get_f32("general.sampling.temp") else {
                continue;
            };
            let family = ModelFamily::detect(&gguf).unwrap();
            let identity = ReleaseIdentity::detect(family, &gguf);
            let decided = release_sampling(identity, 42);
            let declared_top_p = gguf.get_f32("general.sampling.top_p").unwrap_or(1.0);
            let declared_top_k = gguf
                .get_u64("general.sampling.top_k")
                .map_or(0, |k| k as usize);
            let declared_min_p = gguf.get_f32("general.sampling.min_p").unwrap_or(0.0);
            let context = format!("{} as {}", path.display(), identity.label());
            assert!(
                (decided.temperature - declared_temp).abs() < 1e-6,
                "{context}"
            );
            assert!((decided.top_p - declared_top_p).abs() < 1e-6, "{context}");
            assert_eq!(decided.top_k, declared_top_k, "{context}");
            assert!((decided.min_p - declared_min_p).abs() < 1e-6, "{context}");
            checked += 1;
        }
        eprintln!("release sampling cross-checked against {checked} local GGUF header(s)");
    }
}
