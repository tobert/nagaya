//! The block `{% generation %}` of transformers, which the LFM3 chat templates use (spec 11.3, 11.11). transformers
//! adds it to jinja2 to mark the text of an assistant turn (`AssistantTracker`). The block writes its body as it is.
//! The expected text is the output of `_compile_jinja_template` of transformers 5.20.2 with jinja2 3.1.6, run on
//! 2026-11-06.

use nagaya_jinja::{Json, Template};

fn render(source: &str) -> Result<String, String> {
    let template = Template::from_source(source).map_err(|e| e.to_string())?;
    template
        .render(&[("ms", Json::Array(vec![Json::Int(1), Json::Int(2)]))])
        .map_err(|e| e.to_string())
}

#[test]
fn the_generation_block_writes_its_body_as_jinja2_does() {
    let cases = [
        ("a{% generation %}b{% endgeneration %}c", "abc"),
        ("a\n{%- generation -%}\n b \n{%- endgeneration -%}\nc", "abc"),
        (
            "{% for m in ms %}{%- generation %}[{{ m }}]{% endgeneration -%}\n{% endfor %}",
            "[1][2]",
        ),
        ("  {% generation %}\n  in\n  {% endgeneration %}\nout", "  in\nout"),
        (
            "{% generation %}{% if true %}{% generation %}n{% endgeneration %}{% endif %}{% endgeneration %}",
            "n",
        ),
    ];
    for (source, want) in cases {
        assert_eq!(render(source).as_deref(), Ok(want), "{source:?}");
    }
}

#[test]
fn the_word_generation_outside_a_tag_stays_as_it_is() {
    let cases = [
        ("{{ 'generation' }}{# {% generation %} #}", "generation"),
        ("{{ '{% generation %}' }}", "{% generation %}"),
        ("{% raw %}{% generation %}{% endraw %}", "{% generation %}"),
        ("{% set generation = 3 %}{{ generation }}", "3"),
        ("generation endgeneration", "generation endgeneration"),
    ];
    for (source, want) in cases {
        assert_eq!(render(source).as_deref(), Ok(want), "{source:?}");
    }
}

#[test]
fn a_generation_block_that_does_not_close_is_a_syntax_error() {
    let e = render("{% generation %}x").expect_err("the block does not close");
    assert!(e.contains("syntax error"), "{e}");
}
