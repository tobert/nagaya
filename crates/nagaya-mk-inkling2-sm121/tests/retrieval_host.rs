//! The host parts of the retrieval test (milestone 37 step 11): the snapshot files, the needles, the haystack and the
//! check of the prefix. They need no GPU and no model.

mod retrieval_support;

use retrieval_support::haystack::{assemble, check_prefix};
use retrieval_support::needles::{self, Needle};
use retrieval_support::snapshot::{Key, Reader, Writer, token_hash};

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nagaya-retrieval-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn key(format: &str) -> Key {
    Key {
        identity: [7; 32],
        tokens: token_hash(&[1, 2, 3]),
        format: format.to_owned(),
    }
}

fn noise(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn write_two(dir: &std::path::Path, key: &Key) -> u64 {
    let mut writer = Writer::create(dir, key, 99).unwrap();
    writer.note = serde_json::json!({"logits": "abc"});
    writer.add(0, "k", 0, 4, &noise(1000, 1)).unwrap();
    writer.add(1, "conv", 0, 0, &noise(300, 2)).unwrap();
    writer.finish().unwrap()
}

#[test]
fn a_snapshot_reads_back_what_was_written_and_gives_its_facts() {
    let dir = temp("round");
    let size = write_two(&dir, &key("f32"));
    assert_eq!(std::fs::metadata(key("f32").path(&dir)).unwrap().len(), size);
    let mut reader = Reader::open(&dir, &key("f32")).unwrap().expect("the file is there");
    assert_eq!((reader.n, reader.note["logits"].as_str()), (99, Some("abc")));
    let mut buf = Vec::new();
    let a = reader.read_next(&mut buf).unwrap();
    assert_eq!((a.id, a.rows, a.len), (0, 4, 1000));
    assert_eq!(buf, noise(1000, 1));
    reader.read_next(&mut buf).unwrap();
    assert_eq!(buf, noise(300, 2));
    reader.finish().unwrap();
}

#[test]
fn a_snapshot_of_another_key_is_not_found_and_a_renamed_file_is_refused() {
    let dir = temp("key");
    write_two(&dir, &key("f32"));
    assert!(Reader::open(&dir, &key("bf16")).unwrap().is_none());
    // A file under the name of another key holds its own key, and the open refuses it.
    std::fs::rename(key("f32").path(&dir), key("e4m3").path(&dir)).unwrap();
    let error = Reader::open(&dir, &key("e4m3")).err().expect("the open must refuse");
    assert!(error.contains("key inside the file"), "{error}");
}

#[test]
fn a_flipped_byte_in_the_data_fails_the_blake3_check() {
    let dir = temp("flip");
    write_two(&dir, &key("f32"));
    let path = key("f32").path(&dir);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[500] ^= 0x01;
    std::fs::write(&path, bytes).unwrap();
    let mut reader = Reader::open(&dir, &key("f32")).unwrap().unwrap();
    let mut buf = Vec::new();
    reader.read_next(&mut buf).unwrap();
    reader.read_next(&mut buf).unwrap();
    let error = reader.finish().expect_err("the BLAKE3 check must fail");
    assert!(error.contains("BLAKE3"), "{error}");
}

#[test]
fn a_cut_file_and_a_file_of_the_wrong_size_fail_at_the_open() {
    let dir = temp("cut");
    write_two(&dir, &key("f32"));
    let path = key("f32").path(&dir);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 5]).unwrap();
    assert!(Reader::open(&dir, &key("f32")).is_err(), "a cut file must fail");
    // One byte more in the data keeps the end mark but breaks the size.
    let mut longer = bytes.clone();
    longer.insert(10, 0);
    std::fs::write(&path, longer).unwrap();
    let error = Reader::open(&dir, &key("f32")).err().expect("a longer file must fail");
    assert!(error.contains("bytes"), "{error}");
}

#[test]
fn a_writer_that_is_not_finished_leaves_no_file() {
    let dir = temp("unfinished");
    {
        let mut writer = Writer::create(&dir, &key("f32"), 1).unwrap();
        writer.add(0, "k", 0, 1, &noise(10, 3)).unwrap();
    }
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
}

#[test]
fn a_reader_that_stops_early_fails_at_the_end() {
    let dir = temp("early");
    write_two(&dir, &key("f32"));
    let mut reader = Reader::open(&dir, &key("f32")).unwrap().unwrap();
    reader.read_next(&mut Vec::new()).unwrap();
    assert!(reader.finish().is_err());
}

#[test]
fn the_needles_are_the_same_for_a_seed_and_differ_from_each_other() {
    let percents = [1, 5, 10, 25, 50, 75, 90];
    let a = needles::make(42, &percents);
    assert_eq!(a, needles::make(42, &percents));
    assert_ne!(a, needles::make(43, &percents));
    for (i, x) in a.iter().enumerate() {
        for y in &a[i + 1..] {
            assert!(x.label != y.label && x.code_word != y.code_word && x.number != y.number);
        }
        assert_eq!(x.number.len(), 6);
    }
    println!("{:?}", a[0]);
}

#[test]
fn a_reply_is_right_only_when_it_holds_the_code_word_and_the_number() {
    let n: Needle = needles::make(42, &[10]).remove(0);
    assert!(n.right(&format!(
        "The code word is {} and the number is {}.",
        n.code_word, n.number
    )));
    assert!(n.right(&format!("{} / {}", n.code_word.to_uppercase(), n.number)));
    assert!(!n.right(&format!("The code word is {}.", n.code_word)));
    assert!(!n.right(&format!("The number is {}.", n.number)));
    assert!(!n.right(&n.question()), "an echo of the question is not an answer");
    assert!(!n.right("I cannot find that entry."));
}

fn books(n: usize) -> String {
    (0..400).map(|i| format!("book {n} line {i} of plain text\n")).collect()
}

fn count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[test]
fn the_needles_go_into_the_haystack_once_each_at_their_percent() {
    let list = needles::make(1, &[1, 25, 50, 90]);
    let h = assemble(books, count, 20_000, &list);
    for (needle, at) in list.iter().zip(&h.at) {
        assert_eq!(h.text.matches(&needle.sentence()).count(), 1);
        assert!(h.text[*at..].starts_with(&needle.sentence()));
        let share = *at as f64 / h.text.len() as f64 * 100.0;
        assert!(
            (share - needle.percent as f64).abs() < 2.0,
            "the needle of {}% is at {share:.1}%",
            needle.percent
        );
    }
    let tokens = count(&h.text);
    assert!((19_000..21_500).contains(&tokens), "{tokens} words for 20000 wanted");
    let none = assemble(books, count, 20_000, &[]);
    assert!(!none.text.contains("Registry entry"));
}

#[test]
fn the_check_of_the_prefix_names_the_first_token_that_changes() {
    check_prefix(&[1, 2, 3], &[1, 2, 3, 4], "the render").unwrap();
    let error = check_prefix(&[1, 2, 3], &[1, 9, 3, 4], "the render").unwrap_err();
    assert!(error.contains("token 1 differs"), "{error}");
    assert!(check_prefix(&[1, 2, 3], &[1, 2], "the render").is_err());
}
