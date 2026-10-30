//! The task text and the template id (spec 11.10, "The served embedding model").

mod common;

use nagaya_text_embeddinggemma2::TASKS;

/// The task text of spec 11.10, written out from its table.
const TASK_TEXT: &str = "query\ttask: search result | query: \n\
document\ttitle: none | text: \n\
classification\ttask: classification | query: \n\
clustering\ttask: clustering | query: \n\
similarity\ttask: sentence similarity | query: \n\
code_retrieval\ttask: code retrieval | query: \n\
question_answering\ttask: question answering | query: \n\
fact_checking\ttask: fact checking | query: \n\
frame\t<bos>{prompt}{text}<eos>\n";

#[test]
fn the_tasks_are_those_of_the_spec() {
    let tasks: Vec<&str> = TASKS.iter().map(|(task, _)| *task).collect();
    assert_eq!(
        tasks,
        [
            "query",
            "document",
            "classification",
            "clustering",
            "similarity",
            "code_retrieval",
            "question_answering",
            "fact_checking"
        ]
    );
    if common::absent() {
        return;
    }
    assert_eq!(common::family().tasks(), tasks);
}

#[test]
fn the_task_text_lists_each_task_and_the_frame() {
    if common::absent() {
        return;
    }
    assert_eq!(common::family().task_text(), TASK_TEXT);
}

#[test]
fn the_template_id_is_the_hash_of_the_task_text() {
    if common::absent() {
        return;
    }
    use sha2::{Digest, Sha256};
    let hex: String = Sha256::digest(TASK_TEXT.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        common::family().template_id(),
        format!("embeddinggemma2-tasks:{}", &hex[..16])
    );
}
