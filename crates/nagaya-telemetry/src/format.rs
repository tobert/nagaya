//! The stderr line format (spec 9.1).
//!
//! A line is the message, then the fields as `key=value`. A warning or an error ends with `level=warn` or
//! `level=error`. The line has no time, no level prefix and no span names. systemd's journal adds the time, and the
//! tests of the process read lines by the start of the message.

use std::fmt::{self, Write as _};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::registry::LookupSpan;

/// The event format of the nagaya.
#[derive(Clone, Copy, Debug, Default)]
pub struct StderrFormat;

#[derive(Default)]
struct Fields {
    message: String,
    rest: String,
}

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            let _ = write!(self.rest, " {}={value:?}", field.name());
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            let _ = write!(self.rest, " {}={value}", field.name());
        }
    }
}

impl<S, N> FormatEvent<S, N> for StderrFormat
where
    S: Subscriber + for<'span> LookupSpan<'span>,
    N: for<'writer> FormatFields<'writer> + 'static,
{
    fn format_event(&self, _context: &FmtContext<'_, S, N>, mut writer: Writer<'_>, event: &Event<'_>) -> fmt::Result {
        let mut fields = Fields::default();
        event.record(&mut fields);
        write!(writer, "{}{}", fields.message, fields.rest)?;
        match *event.metadata().level() {
            Level::WARN => write!(writer, " level=warn")?,
            Level::ERROR => write!(writer, " level=error")?,
            _ => {}
        }
        writeln!(writer)
    }
}
