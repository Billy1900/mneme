//! In-memory envelope index using brute-force cosine similarity.
//!
//! No external dependencies. Good for development, testing,
//! and small-scale deployments (<10K engrams).
//!
//! For production at scale, use the Qdrant backend instead.

use async_trait::async_trait;
use chrono::Utc;
use mneme_core::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

use crate::{EnvelopeIndex, StoreError, StoreStats};

/// Lowercase, split on non-alphanumeric runs. Shared by indexing and querying
/// so tokenization is always consistent between the two.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;
/// RRF fusion constant — standard choice from the original RRF paper,
/// dampens the influence of any single rank-1 hit from one channel.
const RRF_K: f32 = 60.0;

#[derive(Clone)]
pub struct InMemoryEnvelopeIndex {
    envelopes: Arc<RwLock<HashMap<Uuid, Envelope>>>,
    /// tag -> envelope ids with that tag. Lets `search()` narrow to a
    /// candidate set via AND-intersection before scoring, instead of
    /// scanning every envelope in the store on every query. Without this,
    /// a tag filter like `uid:{user_id}` (used for multi-tenant isolation)
    /// only affects *correctness*, not cost — every user's search would
    /// still pay for every other user's data.
    tag_index: Arc<RwLock<HashMap<String, HashSet<Uuid>>>>,
    /// term -> (envelope id -> term frequency in that doc's indexed text).
    /// Backs the BM25 lexical channel, fused with vector similarity via RRF.
    term_index: Arc<RwLock<HashMap<String, HashMap<Uuid, u32>>>>,
    /// envelope id -> its own term frequencies. Doubles as (a) the source of
    /// truth for unindexing — no need to re-tokenize old text to know what
    /// to remove — and (b) the per-doc length for BM25's normalization term.
    /// Populated from `summary` on every `upsert` (always available, so
    /// every envelope gets baseline lexical coverage); callers with the full
    /// untruncated text (e.g. `/add`, `/remember`, which only ever put the
    /// first ~100 chars in `summary`) should follow up with
    /// `index_full_text` to replace that baseline with real coverage —
    /// otherwise BM25 can only ever match within the truncated summary.
    doc_terms: Arc<RwLock<HashMap<Uuid, HashMap<String, u32>>>>,
}

impl InMemoryEnvelopeIndex {
    pub fn new() -> Self {
        Self {
            envelopes: Arc::new(RwLock::new(HashMap::new())),
            tag_index: Arc::new(RwLock::new(HashMap::new())),
            term_index: Arc::new(RwLock::new(HashMap::new())),
            doc_terms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn len(&self) -> usize {
        self.envelopes.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of every envelope currently stored, for callers that need to
    /// persist the in-memory store to disk (e.g. periodic snapshotting in
    /// `mneme-server`, since this backend otherwise loses all data on
    /// restart). Not part of the `EnvelopeIndex` trait since other backends
    /// (SQLite, Qdrant) are already durable and don't need it.
    pub fn all(&self) -> Vec<Envelope> {
        self.envelopes.read().unwrap().values().cloned().collect()
    }

    /// Re-index `id` for BM25 using `text` in place of whatever was indexed
    /// for it before (typically the truncated `summary` indexed by
    /// `upsert`). Does not touch the envelope itself — call after `upsert`.
    pub async fn index_full_text(&self, id: Uuid, text: &str) {
        let mut term_index = self.term_index.write().unwrap();
        let mut doc_terms = self.doc_terms.write().unwrap();
        Self::index_doc(&mut term_index, &mut doc_terms, id, text);
    }

    fn unindex_doc(
        term_index: &mut HashMap<String, HashMap<Uuid, u32>>,
        doc_terms: &mut HashMap<Uuid, HashMap<String, u32>>,
        id: Uuid,
    ) {
        if let Some(terms) = doc_terms.remove(&id) {
            for term in terms.keys() {
                if let Some(postings) = term_index.get_mut(term) {
                    postings.remove(&id);
                    if postings.is_empty() {
                        term_index.remove(term);
                    }
                }
            }
        }
    }

    fn index_doc(
        term_index: &mut HashMap<String, HashMap<Uuid, u32>>,
        doc_terms: &mut HashMap<Uuid, HashMap<String, u32>>,
        id: Uuid,
        text: &str,
    ) {
        Self::unindex_doc(term_index, doc_terms, id);
        let mut freq: HashMap<String, u32> = HashMap::new();
        for term in tokenize(text) {
            *freq.entry(term).or_insert(0) += 1;
        }
        for (term, tf) in &freq {
            term_index.entry(term.clone()).or_default().insert(id, *tf);
        }
        doc_terms.insert(id, freq);
    }

    /// BM25 score for `query_terms` against every envelope containing at
    /// least one query term, restricted to `candidate_ids` if given.
    fn bm25_scores(
        &self,
        query_terms: &[String],
        candidate_ids: Option<&HashSet<Uuid>>,
    ) -> HashMap<Uuid, f32> {
        let term_index = self.term_index.read().unwrap();
        let doc_terms = self.doc_terms.read().unwrap();
        let n_docs = doc_terms.len().max(1) as f32;
        let avg_doc_len = if doc_terms.is_empty() {
            1.0
        } else {
            doc_terms
                .values()
                .map(|terms| terms.values().sum::<u32>() as f32)
                .sum::<f32>()
                / doc_terms.len() as f32
        };

        let mut scores: HashMap<Uuid, f32> = HashMap::new();
        for term in query_terms {
            let Some(postings) = term_index.get(term) else {
                continue;
            };
            // Standard BM25 IDF, floored at a small positive value so a term
            // present in nearly every doc still contributes rather than
            // going negative and penalizing matches.
            let idf = ((n_docs - postings.len() as f32 + 0.5) / (postings.len() as f32 + 0.5)
                + 1.0)
                .ln()
                .max(1e-6);
            for (&id, &tf) in postings {
                if let Some(ids) = candidate_ids {
                    if !ids.contains(&id) {
                        continue;
                    }
                }
                let dl = doc_terms
                    .get(&id)
                    .map(|terms| terms.values().sum::<u32>())
                    .unwrap_or(1) as f32;
                let tf = tf as f32;
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_doc_len);
                let term_score = idf * (tf * (BM25_K1 + 1.0)) / denom;
                *scores.entry(id).or_insert(0.0) += term_score;
            }
        }
        scores
    }
}

impl Default for InMemoryEnvelopeIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EnvelopeIndex for InMemoryEnvelopeIndex {
    async fn upsert(&self, envelope: &Envelope) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let mut tag_index = self.tag_index.write().unwrap();
        let mut term_index = self.term_index.write().unwrap();
        let mut doc_terms = self.doc_terms.write().unwrap();

        // Re-upserting an existing id can change its tags — drop the old
        // tag associations before adding the new ones, or the index would
        // accumulate stale entries pointing at a now-different tag set.
        if let Some(old) = store.get(&envelope.id) {
            for t in &old.tags {
                if let Some(set) = tag_index.get_mut(t) {
                    set.remove(&envelope.id);
                }
            }
        }
        for t in &envelope.tags {
            tag_index.entry(t.clone()).or_default().insert(envelope.id);
        }
        // Baseline lexical coverage from `summary` (always present). Callers
        // with the full untruncated text should follow up with
        // `index_full_text` — see the `doc_terms` field doc comment.
        Self::index_doc(
            &mut term_index,
            &mut doc_terms,
            envelope.id,
            &envelope.summary,
        );

        store.insert(envelope.id, envelope.clone());
        Ok(())
    }

    async fn search(&self, query: &MemoryQuery) -> Result<Vec<RetrievalResult>, StoreError> {
        let store = self.envelopes.read().unwrap();

        // When tags are given, narrow to the AND-intersection of their
        // candidate sets first — this is what makes a tag filter (e.g.
        // `uid:{user_id}`) actually cheap instead of just correct.
        let candidate_ids: Option<Vec<Uuid>> = if query.tags.is_empty() {
            None
        } else {
            let tag_index = self.tag_index.read().unwrap();
            let mut tags = query.tags.iter();
            let mut candidates: HashSet<Uuid> = match tags.next() {
                Some(first) => tag_index.get(first).cloned().unwrap_or_default(),
                None => HashSet::new(),
            };
            for t in tags {
                if candidates.is_empty() {
                    break;
                }
                let set = tag_index.get(t).cloned().unwrap_or_default();
                candidates = candidates.intersection(&set).copied().collect();
            }
            Some(candidates.into_iter().collect())
        };

        let envelopes_iter: Box<dyn Iterator<Item = &Envelope>> = match &candidate_ids {
            Some(ids) => Box::new(ids.iter().filter_map(|id| store.get(id))),
            None => Box::new(store.values()),
        };

        let mut results: Vec<RetrievalResult> = envelopes_iter
            .filter(|env| {
                if query.active_only && !env.is_active() {
                    return false;
                }
                if let Some(ref mt) = query.memory_type {
                    if env.memory_type != *mt {
                        return false;
                    }
                }
                if let Some(min_conf) = query.min_confidence {
                    if env.confidence < min_conf {
                        return false;
                    }
                }
                true
            })
            .map(|env| {
                let similarity = env.embedding.cosine_similarity(&query.embedding);
                let recency = env.time_decay(0.05) as f32;
                let retrieval_score =
                    (1.0 - query.recency_weight) * similarity + query.recency_weight * recency;
                RetrievalResult {
                    envelope: env.clone(),
                    similarity,
                    retrieval_score,
                }
            })
            .collect();

        results.sort_by(|a, b| b.retrieval_score.partial_cmp(&a.retrieval_score).unwrap());

        // Lexical (BM25) channel, fused with the vector ranking above via
        // Reciprocal Rank Fusion, when the caller supplied query text. This
        // catches keyword/entity matches (names, numbers, exact terms) that
        // pure cosine similarity over embeddings can miss.
        let query_terms = tokenize(&query.query_text);
        if !query_terms.is_empty() && !results.is_empty() {
            let allowed: HashSet<Uuid> = results.iter().map(|r| r.envelope.id).collect();
            let bm25 = self.bm25_scores(&query_terms, Some(&allowed));

            if !bm25.is_empty() {
                let mut bm25_ranked: Vec<(Uuid, f32)> = bm25.into_iter().collect();
                bm25_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let bm25_rank: HashMap<Uuid, usize> = bm25_ranked
                    .iter()
                    .enumerate()
                    .map(|(rank, (id, _))| (*id, rank))
                    .collect();

                let mut fused: Vec<(RetrievalResult, f32)> = results
                    .into_iter()
                    .enumerate()
                    .map(|(vec_rank, r)| {
                        let vec_rrf = 1.0 / (RRF_K + vec_rank as f32 + 1.0);
                        let lex_rrf = bm25_rank
                            .get(&r.envelope.id)
                            .map(|&rank| 1.0 / (RRF_K + rank as f32 + 1.0))
                            .unwrap_or(0.0);
                        (r, vec_rrf + lex_rrf)
                    })
                    .collect();
                fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                results = fused.into_iter().map(|(r, _)| r).collect();
            }
        }

        results.truncate(query.top_k);
        Ok(results)
    }

    async fn get(&self, id: Uuid) -> Result<Envelope, StoreError> {
        let store = self.envelopes.read().unwrap();
        store.get(&id).cloned().ok_or(StoreError::NotFound(id))
    }

    async fn get_batch(&self, ids: &[Uuid]) -> Result<Vec<Envelope>, StoreError> {
        let store = self.envelopes.read().unwrap();
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            match store.get(id) {
                Some(env) => results.push(env.clone()),
                None => return Err(StoreError::NotFound(*id)),
            }
        }
        Ok(results)
    }

    async fn list_working_memory(&self, session_id: &str) -> Result<Vec<Envelope>, StoreError> {
        let store = self.envelopes.read().unwrap();
        let results = store
            .values()
            .filter(|env| {
                env.memory_type == MemoryType::Working
                    && env.source_sessions.iter().any(|s| s == session_id)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    async fn mark_superseded(&self, id: Uuid, successor: Uuid) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let env = store.get_mut(&id).ok_or(StoreError::NotFound(id))?;
        env.superseded_by = Some(successor);
        env.updated_at = Utc::now();
        Ok(())
    }

    async fn gc(&self, confidence_floor: f32, older_than_hours: u64) -> Result<usize, StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let cutoff = Utc::now() - chrono::Duration::hours(older_than_hours as i64);
        let to_remove: Vec<Uuid> = store
            .iter()
            .filter(|(_, env)| {
                let too_old = env.memory_type == MemoryType::Working && env.created_at < cutoff;
                let too_low = env.confidence < confidence_floor && !env.is_active();
                too_old || too_low
            })
            .map(|(id, _)| *id)
            .collect();

        let mut tag_index = self.tag_index.write().unwrap();
        let mut term_index = self.term_index.write().unwrap();
        let mut doc_terms = self.doc_terms.write().unwrap();
        for id in &to_remove {
            if let Some(env) = store.remove(id) {
                for t in &env.tags {
                    if let Some(set) = tag_index.get_mut(t) {
                        set.remove(id);
                    }
                }
                Self::unindex_doc(&mut term_index, &mut doc_terms, *id);
            }
        }
        Ok(to_remove.len())
    }

    async fn delete(&self, id: Uuid) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let env = store.remove(&id).ok_or(StoreError::NotFound(id))?;
        let mut tag_index = self.tag_index.write().unwrap();
        for t in &env.tags {
            if let Some(set) = tag_index.get_mut(t) {
                set.remove(&id);
            }
        }
        let mut term_index = self.term_index.write().unwrap();
        let mut doc_terms = self.doc_terms.write().unwrap();
        Self::unindex_doc(&mut term_index, &mut doc_terms, id);
        Ok(())
    }

    async fn touch(&self, id: Uuid, new_confidence: f32) -> Result<(), StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let env = store.get_mut(&id).ok_or(StoreError::NotFound(id))?;
        env.access_count += 1;
        env.last_accessed_at = Utc::now();
        env.confidence = new_confidence;
        Ok(())
    }

    async fn apply_decay(&self, lambda: f64) -> Result<usize, StoreError> {
        let mut store = self.envelopes.write().unwrap();
        let mut updated = 0;
        for env in store.values_mut() {
            if env.is_active() {
                let new_conf = (env.confidence as f64 * env.time_decay(lambda)) as f32;
                env.confidence = new_conf.max(0.0);
                updated += 1;
            }
        }
        Ok(updated)
    }

    async fn stats(&self) -> Result<StoreStats, StoreError> {
        let store = self.envelopes.read().unwrap();
        let total = store.len();
        let working = store
            .values()
            .filter(|e| e.memory_type == MemoryType::Working)
            .count();
        let semantic = store
            .values()
            .filter(|e| e.memory_type == MemoryType::Semantic)
            .count();
        let superseded = store.values().filter(|e| !e.is_active()).count();
        let avg_confidence = if total == 0 {
            0.0
        } else {
            store.values().map(|e| e.confidence).sum::<f32>() / total as f32
        };
        Ok(StoreStats {
            total_engrams: total,
            working_memory_count: working,
            semantic_memory_count: semantic,
            superseded_count: superseded,
            avg_confidence,
        })
    }
}
