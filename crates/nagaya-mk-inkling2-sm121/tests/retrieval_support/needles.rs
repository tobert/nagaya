//! The needles: synthetic facts with random-looking answers. A seed fixes them, so that each run asks the same
//! questions of the same text.

/// A splitmix64 generator. It is small and it gives the same numbers on each machine.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const CONSONANTS: &[u8] = b"bdfgklmnprstvz";
const VOWELS: &[u8] = b"aeiou";

/// A word of `syllables` syllables: a consonant, then a vowel.
fn word(rng: &mut Rng, syllables: usize) -> String {
    (0..syllables)
        .flat_map(|_| {
            [
                CONSONANTS[rng.below(CONSONANTS.len() as u64) as usize] as char,
                VOWELS[rng.below(VOWELS.len() as u64) as usize] as char,
            ]
        })
        .collect()
}

/// One fact that the context holds, and the question about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Needle {
    /// The place that the needle aims for, in percent of the context.
    pub percent: u32,
    /// The name of the registry entry. The question names it.
    pub label: String,
    pub code_word: String,
    pub number: String,
}

/// The needles of `percents`, from `seed`. Two needles never share a label, a code word or a number.
pub fn make(seed: u64, percents: &[u32]) -> Vec<Needle> {
    let mut rng = Rng(seed);
    let mut out: Vec<Needle> = Vec::new();
    for percent in percents {
        loop {
            let label = format!("{}-{:03}", word(&mut rng, 2).to_uppercase(), rng.below(1000));
            let code_word = format!("{}-{}", word(&mut rng, 3), word(&mut rng, 3));
            let number = format!("{:06}", 100_000 + rng.below(900_000));
            let clash = out
                .iter()
                .any(|n| n.label == label || n.code_word == code_word || n.number == number);
            if !clash {
                out.push(Needle {
                    percent: *percent,
                    label,
                    code_word,
                    number,
                });
                break;
            }
        }
    }
    out
}

impl Needle {
    /// The sentence that goes into the haystack, as its own paragraph.
    pub fn sentence(&self) -> String {
        format!(
            "Registry entry {}: the code word is {} and the access number is {}.",
            self.label, self.code_word, self.number
        )
    }

    /// The question of the later turn.
    pub fn question(&self) -> String {
        format!(
            "In the text above, what are the code word and the access number of registry entry {}? Give both.",
            self.label
        )
    }

    /// The two parts of a right answer, each as a flag: the code word and the access number. The reply must hold
    /// them whole. The question holds neither, so an echo of the question is wrong.
    pub fn check(&self, reply: &str) -> (bool, bool) {
        let lower = reply.to_lowercase();
        (lower.contains(&self.code_word), reply.contains(&self.number))
    }

    /// True when the reply holds both parts.
    pub fn right(&self, reply: &str) -> bool {
        let (word, number) = self.check(reply);
        word && number
    }
}

/// The number of the first book of a context. The first context, with the default seed, starts at book 0. Another seed
/// starts at a book from 10 to 89, so that its text is not the text of the first context.
pub fn first_book(seed: u64, default_seed: u64) -> usize {
    if seed == default_seed {
        0
    } else {
        10 + (seed % 80) as usize
    }
}
