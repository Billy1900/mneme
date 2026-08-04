mod datasets;
mod judge;
mod metrics;
mod runner;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use mneme_consolidate::{AnthropicLLM, DeepSeekLLM, OpenAILLM};
use mneme_embed::backends::{LocalEmbeddingModel, OpenAIEmbeddingModel};
use mneme_embed::EmbeddingModel;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[derive(Debug, Clone, ValueEnum)]
enum Bench {
    Locomo,
    Longmemeval,
}

#[derive(Debug, Clone, ValueEnum)]
enum LlmBackend {
    Deepseek,
    Anthropic,
    Openai,
}

#[derive(Debug, Clone, ValueEnum)]
enum EmbedBackend {
    Local,
    Openai,
}

#[derive(Parser, Debug)]
#[command(name = "mneme-bench", about = "Run LoCoMo or LongMemEval on mneme")]
struct Args {
    /// Which benchmark to run
    #[arg(long, value_enum)]
    bench: Bench,

    /// Path to the converted dataset JSON file
    #[arg(long)]
    data: PathBuf,

    /// Max conversations/items to evaluate (for quick runs)
    #[arg(long)]
    limit: Option<usize>,

    /// Only evaluate questions in this category (e.g. multi-hop, temporal)
    #[arg(long)]
    category: Option<String>,

    /// Number of memories to retrieve per question
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Run LLM-as-judge scoring (costs extra API calls)
    #[arg(long, default_value_t = true)]
    judge: bool,

    /// Output results JSON path
    #[arg(long, default_value = "results/run.json")]
    out: PathBuf,

    /// LLM backend for compaction, judge, rerank, and decomposition
    #[arg(long, value_enum, default_value = "deepseek")]
    llm: LlmBackend,

    /// Embedding backend for semantic recall
    #[arg(long, value_enum, default_value = "local")]
    embed: EmbedBackend,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mneme_benchmark=info".into()),
        )
        .init();

    let args = Args::parse();

    let embed: Arc<dyn EmbeddingModel> = match args.embed {
        EmbedBackend::Local => {
            info!("Using local embedding model (BGE-small, no API key)");
            Arc::new(LocalEmbeddingModel::new().context("failed to load local embedding model")?)
        }
        EmbedBackend::Openai => {
            let key = std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY must be set")?;
            Arc::new(OpenAIEmbeddingModel::new(key))
        }
    };

    // LLM backend drives both compaction synthesis and the judge (answer
    // generation, scoring, reranking, query decomposition).
    let (llm, judge) = match args.llm {
        LlmBackend::Deepseek => {
            let key = std::env::var("DEEPSEEK_API_KEY").context("DEEPSEEK_API_KEY must be set")?;
            info!("Using DeepSeek for compaction LLM and judge");
            (
                runner::ConsolidationLlm::DeepSeek(DeepSeekLLM::new(key.clone())),
                judge::LLMJudge::new_deepseek(key),
            )
        }
        LlmBackend::Anthropic => {
            let claude = AnthropicLLM::from_claude_credentials()
                .map_err(|e| anyhow::anyhow!("Claude credentials unavailable: {e}"))?;
            let key = std::env::var("DEEPSEEK_API_KEY")
                .context("DEEPSEEK_API_KEY must be set (judge runs on DeepSeek when llm=anthropic)")?;
            info!("Using Claude (OAuth) for compaction LLM, DeepSeek for judge");
            (
                runner::ConsolidationLlm::Anthropic(claude),
                judge::LLMJudge::new_deepseek(key),
            )
        }
        LlmBackend::Openai => {
            let key = std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY must be set")?;
            info!("Using OpenAI for compaction LLM and judge");
            (
                runner::ConsolidationLlm::OpenAI(OpenAILLM::new(key.clone())),
                judge::LLMJudge::new(key),
            )
        }
    };

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (results, bench_name) = match args.bench {
        Bench::Locomo => {
            info!("Loading LoCoMo dataset from {:?}", args.data);
            let mut convs = datasets::locomo::load(&args.data)
                .with_context(|| format!("Failed to load {:?}", args.data))?;
            if let Some(cat) = &args.category {
                for c in &mut convs {
                    c.questions.retain(|q| &q.category == cat);
                }
                convs.retain(|c| !c.questions.is_empty());
            }
            info!("Loaded {} conversations", convs.len());

            let results = runner::run_locomo(
                &convs, embed, llm, &judge, args.top_k, args.judge, args.limit,
            )
            .await?;
            (results, "locomo")
        }
        Bench::Longmemeval => {
            info!("Loading LongMemEval dataset from {:?}", args.data);
            let items = datasets::longmemeval::load(&args.data)
                .with_context(|| format!("Failed to load {:?}", args.data))?;
            info!("Loaded {} items", items.len());

            if items.is_empty() {
                bail!("Dataset is empty");
            }

            let results = runner::run_longmemeval(
                &items, embed, llm, &judge, args.top_k, args.judge, args.limit,
            )
            .await?;
            (results, "longmemeval")
        }
    };

    if results.is_empty() {
        bail!("No results produced — check dataset format");
    }

    let embed_name = match args.embed {
        EmbedBackend::Local => "bge-small-en-v1.5",
        EmbedBackend::Openai => "text-embedding-3-small",
    };
    let llm_name = match args.llm {
        LlmBackend::Deepseek => "deepseek-v4-flash",
        LlmBackend::Anthropic => "claude-haiku-4-5 + deepseek-chat (judge)",
        LlmBackend::Openai => "gpt-4o-mini",
    };
    let summary = metrics::aggregate(bench_name, embed_name, llm_name, results);

    // Print summary to stdout
    println!(
        "\n========== {} RESULTS ==========",
        bench_name.to_uppercase()
    );
    println!("Total questions : {}", summary.total_questions);
    println!("Exact match     : {:.1}%", summary.exact_match * 100.0);
    println!("Token F1        : {:.3}", summary.f1);
    if let Some(js) = summary.judge_score {
        println!("Judge score     : {:.3}", js);
    }
    println!("Avg tokens/query: {:.0}", summary.avg_tokens_per_query);
    println!("Latency p50     : {}ms", summary.p50_latency_ms);
    println!("Latency p95     : {}ms", summary.p95_latency_ms);
    println!("\nPer-category breakdown:");
    let mut cats: Vec<_> = summary.per_category.iter().collect();
    cats.sort_by_key(|(k, _)| k.as_str());
    for (cat, m) in &cats {
        print!(
            "  {:30} n={:4}  EM={:.1}%  F1={:.3}",
            cat,
            m.count,
            m.exact_match * 100.0,
            m.f1
        );
        if let Some(js) = m.judge_score {
            print!("  judge={:.3}", js);
        }
        println!();
    }
    println!("=================================\n");

    // Write full results to file
    let json = serde_json::to_string_pretty(&summary)?;
    std::fs::write(&args.out, &json)?;
    info!("Results written to {:?}", args.out);

    Ok(())
}
