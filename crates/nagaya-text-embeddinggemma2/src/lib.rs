//! The EmbeddingGemma 2 family library (spec 11.10): the tokenizer, the task prompts and the template id.
//!
//! The family turns one input and its task into the ids that the engine embeds. The engine never takes text
//! (engine-abi D2). The ids are those of sentence-transformers 6.1.0, which the oracle recorded (spec 8.3).

mod error;

use std::path::Path;

use sha2::{Digest, Sha256};

pub use error::Error;

/// Each task of the served embedding model, in the order of spec 11.10: the task of a request, and the name of its
/// prompt in `config_sentence_transformers.json`.
pub const TASKS: [(&str, &str); 8] = [
    ("query", "query"),
    ("document", "document"),
    ("classification", "Classification"),
    ("clustering", "Clustering"),
    ("similarity", "SentenceSimilarity"),
    ("code_retrieval", "CodeRetrieval"),
    ("question_answering", "QuestionAnswering"),
    ("fact_checking", "FactChecking"),
];

/// The start of the template id (spec 11.10).
pub const TEMPLATE_PREFIX: &str = "embeddinggemma2-tasks:";

/// The family of an EmbeddingGemma 2 checkpoint.
pub struct EmbeddingGemma2Family {
    /// The tokenizer with special-token matching off: text that spells `<eos>` stays text.
    content: tokenizers::Tokenizer,
    bos: u32,
    eos: u32,
    /// The prompt of each task, in the order of `TASKS`.
    prompts: Vec<String>,
    tokenizer_hash: String,
    template_id: String,
}

impl std::fmt::Debug for EmbeddingGemma2Family {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemma2Family")
            .field("tokenizer_hash", &self.tokenizer_hash)
            .field("template_id", &self.template_id)
            .finish()
    }
}

fn read(path: &Path) -> Result<Vec<u8>, Error> {
    std::fs::read(path).map_err(|source| Error::Io {
        path: path.display().to_string(),
        source,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

impl EmbeddingGemma2Family {
    /// Build the family from the checkpoint directory `dir`. It reads `tokenizer.json` and
    /// `config_sentence_transformers.json`.
    pub fn from_dir(dir: &Path) -> Result<EmbeddingGemma2Family, Error> {
        let tokenizer = read(&dir.join("tokenizer.json"))?;
        let config = read(&dir.join("config_sentence_transformers.json"))?;
        EmbeddingGemma2Family::from_bytes(&tokenizer, &config)
    }

    /// Build the family from the bytes of `tokenizer.json` and of `config_sentence_transformers.json`.
    pub fn from_bytes(tokenizer: &[u8], config: &[u8]) -> Result<EmbeddingGemma2Family, Error> {
        let mut content = tokenizers::Tokenizer::from_bytes(tokenizer).map_err(|e| Error::Tokenizer(e.to_string()))?;
        content.set_encode_special_tokens(true);
        let special = |token: &str| {
            content
                .token_to_id(token)
                .ok_or_else(|| Error::Tokenizer(format!("tokenizer.json has no token {token}")))
        };
        let (bos, eos) = (special("<bos>")?, special("<eos>")?);
        let config: serde_json::Value = serde_json::from_slice(config).map_err(|e| Error::Config(e.to_string()))?;
        let table = config["prompts"]
            .as_object()
            .ok_or_else(|| Error::Config("it has no prompts object".to_owned()))?;
        let prompts = TASKS
            .iter()
            .map(|&(task, name)| match table.get(name).and_then(|p| p.as_str()) {
                Some(prompt) => Ok(prompt.to_owned()),
                None => Err(Error::Prompt { task, name }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut family = EmbeddingGemma2Family {
            content,
            bos,
            eos,
            prompts,
            tokenizer_hash: format!("sha256:{}", sha256_hex(tokenizer)),
            template_id: String::new(),
        };
        family.template_id = format!("{TEMPLATE_PREFIX}{}", &sha256_hex(family.task_text().as_bytes())[..16]);
        Ok(family)
    }

    /// The tasks of a request, in the order of spec 11.10.
    pub fn tasks(&self) -> Vec<&'static str> {
        TASKS.iter().map(|(task, _)| *task).collect()
    }

    /// The prompt of `task`. A task that the family does not have is an error.
    pub fn prompt(&self, task: &str) -> Result<&str, Error> {
        TASKS
            .iter()
            .position(|(name, _)| *name == task)
            .map(|index| self.prompts[index].as_str())
            .ok_or_else(|| Error::Task { task: task.to_owned() })
    }

    /// The ids of one input: `<bos>`, the ids of the prompt and the text as one string, and `<eos>`.
    pub fn ids(&self, task: &str, text: &str) -> Result<Vec<u32>, Error> {
        let joined = format!("{}{text}", self.prompt(task)?);
        let encoding = self
            .content
            .encode(joined, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids = Vec::with_capacity(encoding.get_ids().len() + 2);
        ids.push(self.bos);
        ids.extend_from_slice(encoding.get_ids());
        ids.push(self.eos);
        Ok(ids)
    }

    /// The token count of one input, with its prompt and its special tokens. A dry run gives it.
    pub fn tokens(&self, task: &str, text: &str) -> Result<usize, Error> {
        Ok(self.ids(task, text)?.len())
    }

    /// The tokenizer hash of the identity: `sha256:` and the hex digits of the SHA-256 of `tokenizer.json`.
    pub fn tokenizer_hash(&self) -> &str {
        &self.tokenizer_hash
    }

    /// The task text: a line for each task, a tab and its prompt, then the line of the frame (spec 11.10).
    pub fn task_text(&self) -> String {
        let mut text = String::new();
        for ((task, _), prompt) in TASKS.iter().zip(&self.prompts) {
            text.push_str(&format!("{task}\t{prompt}\n"));
        }
        text.push_str("frame\t<bos>{prompt}{text}<eos>\n");
        text
    }

    /// The template id of the identity: `embeddinggemma2-tasks:` and the first 16 hex digits of the SHA-256 of the
    /// task text.
    pub fn template_id(&self) -> &str {
        &self.template_id
    }
}
