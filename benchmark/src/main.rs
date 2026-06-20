mod datasets;
mod judge;
mod metrics;
mod runner;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use mneme_consolidate::OpenAILLM;
use mneme_embed::backends::OpenAIEmbeddingModel;
use std::path::PathBuf;
use tracing::info;

#[derive(Debug, Clone, ValueEnum)]
enum Bench {
    Locomo,
    Longmemeval,
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

    /// Number of memories to retrieve per question
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Run LLM-as-judge scoring (costs extra API calls)
    #[arg(long, default_value_t = true)]
    judge: bool,

    /// Output results JSON path
    #[arg(long, default_value = "results/run.json")]
    out: PathBuf,
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

    let openai_key = std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY must be set")?;

    let embed = OpenAIEmbeddingModel::new(openai_key.clone());
    let llm = OpenAILLM::new(openai_key.clone());
    let judge = judge::LLMJudge::new(openai_key.clone());

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (results, bench_name) = match args.bench {
        Bench::Locomo => {
            info!("Loading LoCoMo dataset from {:?}", args.data);
            let convs = datasets::locomo::load(&args.data)
                .with_context(|| format!("Failed to load {:?}", args.data))?;
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

    let summary = metrics::aggregate(
        bench_name,
        "text-embedding-3-small",
        "gpt-4o-mini",
        results,
    );

    // Print summary to stdout
    println!("\n========== {} RESULTS ==========", bench_name.to_uppercase());
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
        print!("  {:30} n={:4}  EM={:.1}%  F1={:.3}", cat, m.count, m.exact_match * 100.0, m.f1);
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
