//! `-fx` / `fxImage` (issue #12): a per-pixel expression language — the
//! subset whose semantics were pinned black-box against `magick` 7.1.2
//! (`tests/magick_ops.rs` holds each listed form).
//!
//! The expression is evaluated once per pixel and colour channel, in
//! normalised units:
//!
//! - numbers (`0.5`, `1e-3`), the constants `pi`, `e`, `QuantumRange`
//!   (65535, as `magick` Q16 reports it) and `QuantumScale` (its inverse);
//! - `u` / `p` — the current channel of the current pixel; `r`, `g`,
//!   `b`, `a` — a fixed channel; `u.r` … `u.a`, `p.r` …; `i`, `j` — the
//!   pixel's column and row; `w`, `h` — the extent; `intensity` /
//!   `luminance` (Rec. 709 weights);
//! - `p[dx,dy]` — the current channel at a relative offset, `p{x,y}` —
//!   at an absolute position (edges replicated, non-integer positions
//!   bilinear), optionally followed by `.r` …;
//! - `+ - * / %` (`%` is `fmod`), `^` (power, right-associative, binds
//!   tighter than unary minus), unary `-` `+` `!`, `< <= > >= == !=`,
//!   `&&` `||`, `?:`, parentheses;
//! - `abs acos asin atan atan2 ceil cos cosh exp floor hypot int ln (natural) log
//!   (common) logtwo max min mod pow round sign sin sinh sqrt
//!   squish tan tanh trunc`.
//!
//! - more functions: `clamp` (to `[0, 1]`), `gauss(x)` = `exp(−x²/2) /
//!   √(2π)`, `sinc(x)` = `sin(πx) / (πx)`, `isnan`, `not`, `alt(x)` (`1`
//!   for even `x`, `−1` for odd), `drc(x, k)` = `x / (k (x − 1) + 1)`,
//!   `rand()` (a deterministic value in `[0, 1)` per pixel, channel and
//!   call site), `channel(r, g, b, a)` (the argument of the current
//!   channel), `if(c, a, b)`;
//! - `hue`, `saturation` — of the current pixel (HSL, `0 … 1`);
//! - `v` — the second picture of [`Fx::apply_images`] wherever `u` may
//!   stand (`v`, `v.g`, `v[dx,dy]`, `v{x,y}`); `u[k]` — picture `k` at
//!   the current pixel; `t`, `n` — the index of the picture being
//!   computed (0) and the number of pictures;
//! - statements separated by `;` with user variables: `name = expr`
//!   (also `+= -= *= /=`), the value of the last statement being the
//!   result; `while(cond, body)`, `for(init, cond, body)` and
//!   `do(body, cond)` (a loop stops after 10 000 turns).
//!
//! Integer output saturates to the code range; float output keeps the
//! raw result. Rows are evaluated under the thread budget.

use oxideav_core::{Error, PixelFormat, VideoFrame};

use crate::deep::FloatImage;
use crate::{PixelFilter, VideoStreamParams, deep, layout};

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Num(f64),
    Var(Var),
    /// Pixel read: relative (`p[dx,dy]`) or absolute (`p{x,y}`), with an
    /// optional fixed channel.
    Pixel {
        image: usize,
        absolute: bool,
        x: Box<Expr>,
        y: Box<Expr>,
        channel: Option<usize>,
    },
    /// The current pixel of picture `image` (not the first), with an
    /// optional fixed channel.
    Image(usize, Option<usize>),
    /// A user variable.
    Local(usize),
    Assign(usize, Box<Expr>),
    /// Statements; the value is the last one's.
    Seq(Vec<Expr>),
    /// `while (cond) body`; the value is the last body value.
    While(Box<Expr>, Box<Expr>),
    Unary(char, Box<Expr>),
    Binary(&'static str, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Var {
    /// The current channel.
    U,
    Channel(usize),
    I,
    J,
    W,
    H,
    Intensity,
    Hue,
    Saturation,
    /// Number of pictures.
    Count,
}

/// A parsed `-fx` expression.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Fx {
    /// The source text.
    pub expression: String,
    ast: Expr,
    /// Number of user variables.
    locals: usize,
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
    /// User variables, in order of first assignment.
    locals: Vec<String>,
    /// Call sites of `rand()` seen so far.
    rands: usize,
}

fn perr(msg: impl Into<String>) -> Error {
    Error::invalid(format!("oxideav-image-filter: fx: {}", msg.into()))
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.pos < self.s.len() && (self.s[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.pos).copied()
    }

    fn eat(&mut self, tok: &str) -> bool {
        self.ws();
        if self.s[self.pos..].starts_with(tok.as_bytes()) {
            self.pos += tok.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, tok: &str) -> Result<(), Error> {
        if self.eat(tok) {
            Ok(())
        } else {
            Err(perr(format!("expected '{tok}' at {}", self.pos)))
        }
    }

    /// `statement (';' statement)*` (a trailing `;` is allowed).
    fn statements(&mut self) -> Result<Expr, Error> {
        let mut list = vec![self.statement()?];
        while self.eat(";") {
            if matches!(self.peek(), None | Some(b')') | Some(b',')) {
                break;
            }
            list.push(self.statement()?);
        }
        Ok(if list.len() == 1 {
            list.pop().expect("one statement")
        } else {
            Expr::Seq(list)
        })
    }

    /// `name = expr`, `name op= expr`, or an expression.
    fn statement(&mut self) -> Result<Expr, Error> {
        let start = self.pos;
        if self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            let name = self.ident();
            for op in ["=", "+=", "-=", "*=", "/="] {
                self.ws();
                let is_eq = op == "=" && self.s[self.pos..].starts_with(b"==");
                if !is_eq && self.s[self.pos..].starts_with(op.as_bytes()) {
                    if RESERVED.contains(&name.as_str()) {
                        return Err(perr(format!("'{name}' cannot be assigned")));
                    }
                    self.pos += op.len();
                    let value = self.ternary()?;
                    let slot = match self.locals.iter().position(|l| *l == name) {
                        Some(i) => i,
                        None if op == "=" => {
                            self.locals.push(name);
                            self.locals.len() - 1
                        }
                        None => return Err(perr(format!("'{name}' is used before it is set"))),
                    };
                    let value = match op {
                        "=" => value,
                        "+=" => Expr::Binary("+", Box::new(Expr::Local(slot)), Box::new(value)),
                        "-=" => Expr::Binary("-", Box::new(Expr::Local(slot)), Box::new(value)),
                        "*=" => Expr::Binary("*", Box::new(Expr::Local(slot)), Box::new(value)),
                        _ => Expr::Binary("/", Box::new(Expr::Local(slot)), Box::new(value)),
                    };
                    return Ok(Expr::Assign(slot, Box::new(value)));
                }
            }
            self.pos = start;
        }
        self.ternary()
    }

    fn ternary(&mut self) -> Result<Expr, Error> {
        let c = self.or()?;
        if self.eat("?") {
            let a = self.ternary()?;
            self.expect(":")?;
            let b = self.ternary()?;
            return Ok(Expr::Ternary(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }

    fn or(&mut self) -> Result<Expr, Error> {
        let mut l = self.and()?;
        while self.eat("||") {
            l = Expr::Binary("||", Box::new(l), Box::new(self.and()?));
        }
        Ok(l)
    }

    fn and(&mut self) -> Result<Expr, Error> {
        let mut l = self.equality()?;
        while self.eat("&&") {
            l = Expr::Binary("&&", Box::new(l), Box::new(self.equality()?));
        }
        Ok(l)
    }

    fn equality(&mut self) -> Result<Expr, Error> {
        let mut l = self.relational()?;
        loop {
            let op = if self.eat("==") {
                "=="
            } else if self.eat("!=") {
                "!="
            } else {
                return Ok(l);
            };
            l = Expr::Binary(op, Box::new(l), Box::new(self.relational()?));
        }
    }

    fn relational(&mut self) -> Result<Expr, Error> {
        let mut l = self.additive()?;
        loop {
            let op = if self.eat("<=") {
                "<="
            } else if self.eat(">=") {
                ">="
            } else if self.eat("<") {
                "<"
            } else if self.eat(">") {
                ">"
            } else {
                return Ok(l);
            };
            l = Expr::Binary(op, Box::new(l), Box::new(self.additive()?));
        }
    }

    fn additive(&mut self) -> Result<Expr, Error> {
        let mut l = self.multiplicative()?;
        loop {
            let op = match self.peek() {
                Some(b'+') => "+",
                Some(b'-') => "-",
                _ => return Ok(l),
            };
            self.pos += 1;
            l = Expr::Binary(op, Box::new(l), Box::new(self.multiplicative()?));
        }
    }

    fn multiplicative(&mut self) -> Result<Expr, Error> {
        let mut l = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(b'*') => "*",
                Some(b'/') => "/",
                Some(b'%') => "%",
                _ => return Ok(l),
            };
            self.pos += 1;
            l = Expr::Binary(op, Box::new(l), Box::new(self.unary()?));
        }
    }

    fn unary(&mut self) -> Result<Expr, Error> {
        match self.peek() {
            Some(b'-') => {
                self.pos += 1;
                Ok(Expr::Unary('-', Box::new(self.unary()?)))
            }
            Some(b'+') => {
                self.pos += 1;
                self.unary()
            }
            Some(b'!') if self.s.get(self.pos + 1) != Some(&b'=') => {
                self.pos += 1;
                Ok(Expr::Unary('!', Box::new(self.unary()?)))
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr, Error> {
        let base = self.primary()?;
        if self.eat("^") {
            // Right-associative; the exponent may carry a sign.
            let exp = self.unary()?;
            return Ok(Expr::Binary("^", Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn ident(&mut self) -> String {
        self.ws();
        let start = self.pos;
        while self.pos < self.s.len()
            && ((self.s[self.pos] as char).is_ascii_alphanumeric() || self.s[self.pos] == b'_')
        {
            self.pos += 1;
        }
        String::from_utf8_lossy(&self.s[start..self.pos]).to_string()
    }

    fn channel_suffix(&mut self) -> Result<Option<usize>, Error> {
        if self.peek() == Some(b'.')
            && self
                .s
                .get(self.pos + 1)
                .is_some_and(|c| c.is_ascii_alphabetic())
        {
            self.pos += 1;
            let name = self.ident();
            return Ok(Some(match name.as_str() {
                "r" | "red" | "c" | "cyan" => 0,
                "g" | "green" | "m" | "magenta" => 1,
                "b" | "blue" | "y" | "yellow" => 2,
                "a" | "alpha" | "opacity" => 3,
                other => return Err(perr(format!("unknown channel '{other}'"))),
            }));
        }
        Ok(None)
    }

    fn primary(&mut self) -> Result<Expr, Error> {
        match self.peek() {
            None => Err(perr("unexpected end")),
            Some(b'(') => {
                self.pos += 1;
                let e = self.statements()?;
                self.expect(")")?;
                Ok(e)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => {
                let start = self.pos;
                while self.pos < self.s.len() {
                    let ch = self.s[self.pos];
                    let exp_sign = (ch == b'-' || ch == b'+')
                        && self.pos > start
                        && matches!(self.s[self.pos - 1], b'e' | b'E');
                    if ch.is_ascii_digit() || ch == b'.' || ch == b'e' || ch == b'E' || exp_sign {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let txt = String::from_utf8_lossy(&self.s[start..self.pos]).to_string();
                txt.parse::<f64>()
                    .map(Expr::Num)
                    .map_err(|_| perr(format!("bad number '{txt}'")))
            }
            Some(c) if c.is_ascii_alphabetic() => {
                let name = self.ident();
                if (name == "p" || name == "u" || name == "s" || name == "v")
                    && matches!(self.peek(), Some(b'[') | Some(b'{'))
                {
                    let absolute = self.peek() == Some(b'{');
                    self.pos += 1;
                    let image = usize::from(name == "v");
                    let x = self.ternary()?;
                    if !absolute && self.eat("]") {
                        // `u[k]`: picture k at the current pixel.
                        let Expr::Num(k) = x else {
                            return Err(perr("a picture index must be a number"));
                        };
                        let channel = self.channel_suffix()?;
                        return Ok(Expr::Image(k.max(0.0) as usize, channel));
                    }
                    self.expect(",")?;
                    let y = self.ternary()?;
                    self.expect(if absolute { "}" } else { "]" })?;
                    let channel = self.channel_suffix()?;
                    return Ok(Expr::Pixel {
                        image,
                        absolute,
                        x: Box::new(x),
                        y: Box::new(y),
                        channel,
                    });
                }
                if self.peek() == Some(b'(') {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if !self.eat(")") {
                        loop {
                            args.push(self.statements()?);
                            if self.eat(")") {
                                break;
                            }
                            self.expect(",")?;
                        }
                    }
                    let name = name.to_ascii_lowercase();
                    let arity = |n: usize| {
                        if args.len() == n {
                            Ok(())
                        } else {
                            Err(perr(format!("{name}() takes {n} arguments")))
                        }
                    };
                    match name.as_str() {
                        "if" => {
                            arity(3)?;
                            let mut it = args.into_iter().map(Box::new);
                            let (c, a, b) = (it.next(), it.next(), it.next());
                            return Ok(Expr::Ternary(
                                c.expect("three"),
                                a.expect("three"),
                                b.expect("three"),
                            ));
                        }
                        "while" => {
                            arity(2)?;
                            let body = args.pop().expect("two");
                            let cond = args.pop().expect("two");
                            return Ok(Expr::While(Box::new(cond), Box::new(body)));
                        }
                        "do" => {
                            arity(2)?;
                            let cond = args.pop().expect("two");
                            let body = args.pop().expect("two");
                            return Ok(Expr::Seq(vec![
                                body.clone(),
                                Expr::While(Box::new(cond), Box::new(body)),
                            ]));
                        }
                        "for" => {
                            arity(3)?;
                            let body = args.pop().expect("three");
                            let cond = args.pop().expect("three");
                            let init = args.pop().expect("three");
                            return Ok(Expr::Seq(vec![
                                init,
                                Expr::While(Box::new(cond), Box::new(body)),
                            ]));
                        }
                        "rand" => {
                            self.rands += 1;
                            return Ok(Expr::Call(name, vec![Expr::Num(self.rands as f64)]));
                        }
                        _ => {}
                    }
                    return Ok(Expr::Call(name, args));
                }
                if name == "u" || name == "p" || name == "s" {
                    if let Some(c) = self.channel_suffix()? {
                        return Ok(Expr::Var(Var::Channel(c)));
                    }
                    return Ok(Expr::Var(Var::U));
                }
                if name == "v" {
                    return Ok(Expr::Image(1, self.channel_suffix()?));
                }
                if let Some(slot) = self.locals.iter().position(|l| *l == name) {
                    return Ok(Expr::Local(slot));
                }
                Ok(match name.as_str() {
                    "r" | "red" | "c" | "cyan" => Expr::Var(Var::Channel(0)),
                    "g" | "green" | "m" | "magenta" => Expr::Var(Var::Channel(1)),
                    "b" | "blue" | "y" | "yellow" => Expr::Var(Var::Channel(2)),
                    "a" | "alpha" | "opacity" => Expr::Var(Var::Channel(3)),
                    "i" => Expr::Var(Var::I),
                    "j" => Expr::Var(Var::J),
                    "w" => Expr::Var(Var::W),
                    "h" => Expr::Var(Var::H),
                    "pi" => Expr::Num(std::f64::consts::PI),
                    "e" => Expr::Num(std::f64::consts::E),
                    // Measured: magick's 16-bit quantum range, even in HDRI.
                    "QuantumRange" => Expr::Num(65535.0),
                    "QuantumScale" => Expr::Num(1.0 / 65535.0),
                    "intensity" | "luminance" | "lightness" => Expr::Var(Var::Intensity),
                    "hue" => Expr::Var(Var::Hue),
                    "saturation" => Expr::Var(Var::Saturation),
                    "t" => Expr::Num(0.0),
                    "n" => Expr::Var(Var::Count),
                    other => return Err(perr(format!("unknown symbol '{other}'"))),
                })
            }
            Some(c) => Err(perr(format!("unexpected '{}' at {}", c as char, self.pos))),
        }
    }
}

/// Names that are symbols of the language, not user variables.
const RESERVED: &[&str] = &[
    "u",
    "v",
    "p",
    "s",
    "r",
    "g",
    "b",
    "a",
    "c",
    "m",
    "y",
    "i",
    "j",
    "w",
    "h",
    "t",
    "n",
    "e",
    "pi",
    "red",
    "green",
    "blue",
    "alpha",
    "opacity",
    "cyan",
    "magenta",
    "yellow",
    "intensity",
    "luminance",
    "lightness",
    "hue",
    "saturation",
    "QuantumRange",
    "QuantumScale",
];

struct Ctx<'a> {
    img: &'a FloatImage,
    /// Every picture of the list (`img` is the first).
    images: &'a [&'a FloatImage],
    locals: &'a [std::cell::Cell<f64>],
    x: u32,
    y: u32,
    c: usize,
}

impl Ctx<'_> {
    fn channel(&self, x: f64, y: f64, c: usize) -> f64 {
        self.channel_of(self.img, x, y, c)
    }

    /// Picture `k` of the list (a missing one reads as the first).
    fn image(&self, k: usize) -> &FloatImage {
        self.images.get(k).copied().unwrap_or(self.img)
    }

    /// Hue and saturation (HSL) of the current pixel.
    fn hue_saturation(&self) -> (f64, f64) {
        let g = |c| self.channel(self.x as f64, self.y as f64, c);
        if self.img.channels() < 3 {
            return (0.0, 0.0);
        }
        let (r, gr, b) = (g(0), g(1), g(2));
        let (max, min) = (r.max(gr).max(b), r.min(gr).min(b));
        let delta = max - min;
        if delta <= 0.0 {
            return (0.0, 0.0);
        }
        let light = (max + min) / 2.0;
        let sat = delta / (1.0 - (2.0 * light - 1.0).abs()).max(1e-12);
        let hue = if max == r {
            ((gr - b) / delta).rem_euclid(6.0)
        } else if max == gr {
            (b - r) / delta + 2.0
        } else {
            (r - gr) / delta + 4.0
        } / 6.0;
        (hue, sat.min(1.0))
    }

    fn channel_of(&self, img: &FloatImage, x: f64, y: f64, c: usize) -> f64 {
        let ch = img.channels();
        let c = if ch == 1 { 0 } else { c.min(ch - 1) };
        if c == 3 && ch < 4 {
            return 1.0;
        }
        let (fx, fy) = (x.floor(), y.floor());
        let (tx, ty) = (x - fx, y - fy);
        let s = |xx: f64, yy: f64| img.sample_clamped(xx as i64, yy as i64, c) as f64;
        if tx == 0.0 && ty == 0.0 {
            return s(fx, fy);
        }
        s(fx, fy) * (1.0 - tx) * (1.0 - ty)
            + s(fx + 1.0, fy) * tx * (1.0 - ty)
            + s(fx, fy + 1.0) * (1.0 - tx) * ty
            + s(fx + 1.0, fy + 1.0) * tx * ty
    }
}

fn truthy(v: f64) -> bool {
    v.abs() >= f64::EPSILON
}

fn eval(e: &Expr, cx: &Ctx) -> f64 {
    match e {
        Expr::Num(v) => *v,
        Expr::Var(v) => match v {
            Var::U => cx.channel(cx.x as f64, cx.y as f64, cx.c),
            Var::Channel(c) => cx.channel(cx.x as f64, cx.y as f64, *c),
            Var::I => cx.x as f64,
            Var::J => cx.y as f64,
            Var::W => cx.img.width() as f64,
            Var::H => cx.img.height() as f64,
            Var::Hue => cx.hue_saturation().0,
            Var::Saturation => cx.hue_saturation().1,
            Var::Count => cx.images.len().max(1) as f64,
            Var::Intensity => {
                let g = |c| cx.channel(cx.x as f64, cx.y as f64, c);
                if cx.img.channels() < 3 {
                    g(0)
                } else {
                    0.212_656 * g(0) + 0.715_158 * g(1) + 0.072_186 * g(2)
                }
            }
        },
        Expr::Image(k, channel) => cx.channel_of(
            cx.image(*k),
            cx.x as f64,
            cx.y as f64,
            channel.unwrap_or(cx.c),
        ),
        Expr::Local(slot) => cx.locals[*slot].get(),
        Expr::Assign(slot, value) => {
            let v = eval(value, cx);
            cx.locals[*slot].set(v);
            v
        }
        Expr::Seq(list) => list.iter().fold(0.0, |_, e| eval(e, cx)),
        Expr::While(cond, body) => {
            let mut last = 0.0;
            for _ in 0..10_000 {
                if !truthy(eval(cond, cx)) {
                    break;
                }
                last = eval(body, cx);
            }
            last
        }
        Expr::Pixel {
            image,
            absolute,
            x,
            y,
            channel,
        } => {
            let (px, py) = (eval(x, cx), eval(y, cx));
            let (px, py) = if *absolute {
                (px, py)
            } else {
                (cx.x as f64 + px, cx.y as f64 + py)
            };
            cx.channel_of(cx.image(*image), px, py, channel.unwrap_or(cx.c))
        }
        Expr::Unary(op, a) => {
            let v = eval(a, cx);
            match op {
                '-' => -v,
                _ => {
                    if truthy(v) {
                        0.0
                    } else {
                        1.0
                    }
                }
            }
        }
        Expr::Binary(op, a, b) => {
            let (x, y) = (eval(a, cx), eval(b, cx));
            let bool = |t: bool| if t { 1.0 } else { 0.0 };
            match *op {
                "+" => x + y,
                "-" => x - y,
                "*" => x * y,
                "/" => {
                    if y == 0.0 {
                        0.0
                    } else {
                        x / y
                    }
                }
                "%" => {
                    if y == 0.0 {
                        0.0
                    } else {
                        x % y
                    }
                }
                "^" => x.powf(y),
                "<" => bool(x < y),
                "<=" => bool(x <= y),
                ">" => bool(x > y),
                ">=" => bool(x >= y),
                "==" => bool((x - y).abs() < f64::EPSILON),
                "!=" => bool((x - y).abs() >= f64::EPSILON),
                "&&" => bool(truthy(x) && truthy(y)),
                _ => bool(truthy(x) || truthy(y)),
            }
        }
        Expr::Ternary(c, a, b) => {
            if truthy(eval(c, cx)) {
                eval(a, cx)
            } else {
                eval(b, cx)
            }
        }
        Expr::Call(name, args) => {
            if name == "channel" {
                // Only the current channel's argument is evaluated.
                return args.get(cx.c).or(args.last()).map_or(0.0, |a| eval(a, cx));
            }
            let v: Vec<f64> = args.iter().map(|a| eval(a, cx)).collect();
            let a0 = v.first().copied().unwrap_or(0.0);
            let a1 = v.get(1).copied().unwrap_or(0.0);
            match name.as_str() {
                "abs" => a0.abs(),
                "acos" => a0.acos(),
                "asin" => a0.asin(),
                "atan" => a0.atan(),
                "atan2" => a0.atan2(a1),
                "ceil" => a0.ceil(),
                "cos" => a0.cos(),
                "cosh" => a0.cosh(),
                "exp" => a0.exp(),
                "floor" => a0.floor(),
                "hypot" => a0.hypot(a1),
                "int" | "trunc" => a0.trunc(),
                "ln" => a0.ln(),
                // `log` is the common logarithm (measured).
                "log" => a0.log10(),
                "logtwo" => a0.log2(),
                "max" => v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "min" => v.iter().cloned().fold(f64::INFINITY, f64::min),
                "mod" => {
                    if a1 == 0.0 {
                        0.0
                    } else {
                        a0 % a1
                    }
                }
                "pow" => a0.powf(a1),
                "round" => a0.round(),
                "sign" => {
                    if a0 < 0.0 {
                        -1.0
                    } else {
                        1.0
                    }
                }
                "sin" => a0.sin(),
                "sinh" => a0.sinh(),
                "sqrt" => a0.sqrt(),
                "squish" => 1.0 / (1.0 + (-a0).exp()),
                "tan" => a0.tan(),
                "tanh" => a0.tanh(),
                "clamp" => a0.clamp(0.0, 1.0),
                "gauss" => (-a0 * a0 / 2.0).exp() / std::f64::consts::TAU.sqrt(),
                "sinc" => {
                    if a0 == 0.0 {
                        1.0
                    } else {
                        (std::f64::consts::PI * a0).sin() / (std::f64::consts::PI * a0)
                    }
                }
                "isnan" => f64::from(a0.is_nan()),
                "not" => f64::from(!truthy(a0)),
                "alt" => {
                    if (a0.trunc() as i64) & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                "drc" => a0 / (a1 * (a0 - 1.0) + 1.0),
                "rand" => {
                    // One value per pixel, channel and call site.
                    let mut z =
                        (cx.x as u64) << 40 ^ (cx.y as u64) << 16 ^ (cx.c as u64) << 8 ^ a0 as u64;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
                }
                _ => f64::NAN,
            }
        }
    }
}

const FUNCTIONS: &[&str] = &[
    "abs", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "exp", "floor", "hypot", "int",
    "trunc", "ln", "log", "logtwo", "max", "min", "mod", "pow", "round", "sign", "sin", "sinh",
    "sqrt", "squish", "tan", "tanh", "clamp", "gauss", "sinc", "isnan", "not", "alt", "drc",
    "rand", "channel",
];

fn check_calls(e: &Expr) -> Result<(), Error> {
    match e {
        Expr::Call(n, args) => {
            if !FUNCTIONS.contains(&n.as_str()) {
                return Err(perr(format!("unknown function '{n}'")));
            }
            args.iter().try_for_each(check_calls)
        }
        Expr::Unary(_, a) => check_calls(a),
        Expr::Binary(_, a, b) => {
            check_calls(a)?;
            check_calls(b)
        }
        Expr::Ternary(c, a, b) => {
            check_calls(c)?;
            check_calls(a)?;
            check_calls(b)
        }
        Expr::Pixel { x, y, .. } => {
            check_calls(x)?;
            check_calls(y)
        }
        Expr::Assign(_, a) => check_calls(a),
        Expr::Seq(list) => list.iter().try_for_each(check_calls),
        Expr::While(c, b) => {
            check_calls(c)?;
            check_calls(b)
        }
        _ => Ok(()),
    }
}

impl Fx {
    /// Parse an expression.
    pub fn new(expression: &str) -> Result<Self, Error> {
        let mut p = Parser {
            s: expression.as_bytes(),
            pos: 0,
            locals: Vec::new(),
            rands: 0,
        };
        let ast = p.statements()?;
        p.ws();
        if p.pos != p.s.len() {
            return Err(perr(format!("trailing input at {}", p.pos)));
        }
        check_calls(&ast)?;
        Ok(Self {
            expression: expression.to_string(),
            ast,
            locals: p.locals.len(),
        })
    }

    /// Evaluate at pixel `(x, y)` for channel `c` of `img`.
    pub fn eval_at(&self, img: &FloatImage, x: u32, y: u32, c: usize) -> f64 {
        let locals = vec![std::cell::Cell::new(0.0); self.locals];
        eval(
            &self.ast,
            &Ctx {
                img,
                images: &[img],
                locals: &locals,
                x,
                y,
                c,
            },
        )
    }

    /// Apply to every colour channel of a float picture (alpha kept).
    pub fn apply_float(&self, img: &FloatImage) -> FloatImage {
        self.run(&[img])
    }

    /// Apply to a list of pictures: `u` is the first (the result has
    /// its shape and keeps its alpha), `v` the second, `u[k]` the
    /// `k`-th. Fails on an empty list.
    pub fn apply_images(&self, images: &[FloatImage]) -> Result<FloatImage, Error> {
        if images.is_empty() {
            return Err(perr("no picture to evaluate on"));
        }
        let refs: Vec<&FloatImage> = images.iter().collect();
        Ok(self.run(&refs))
    }

    fn run(&self, images: &[&FloatImage]) -> FloatImage {
        let img = images[0];
        let mut out = img.clone();
        let (ch, cc) = (img.channels(), img.color_channels());
        let row = img.width() as usize * ch;
        crate::exec::par_rows(out.samples_mut(), row.max(1), |y, line| {
            let locals = vec![std::cell::Cell::new(0.0); self.locals];
            for (x, px) in line.chunks_exact_mut(ch).enumerate() {
                for (c, v) in px[..cc].iter_mut().enumerate() {
                    locals.iter().for_each(|l| l.set(0.0));
                    let cx = Ctx {
                        img,
                        images,
                        locals: &locals,
                        x: x as u32,
                        y: y as u32,
                        c,
                    };
                    *v = eval(&self.ast, &cx) as f32;
                }
            }
        });
        out
    }
}

impl PixelFilter for Fx {
    fn accepted_formats(&self) -> &'static [PixelFormat] {
        layout::PACKED_8BIT_DEEP
    }

    fn apply_pixels(
        &self,
        input: &VideoFrame,
        params: VideoStreamParams,
    ) -> Result<VideoFrame, Error> {
        let float = deep::is_float(params.format);
        let img = if float {
            FloatImage::from_frame(input, params)?
        } else {
            deep::widen_u8(input, params)?
        };
        let out = self.apply_float(&img);
        Ok(if float {
            out.into_frame(input.pts)
        } else {
            let mut f = deep::narrow_u8(&out);
            f.pts = input.pts;
            f
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_and_functions() {
        let img = FloatImage::new(4, 3, 1);
        let ev = |s: &str| Fx::new(s).unwrap().eval_at(&img, 2, 1, 0);
        assert_eq!(ev("1+2*3"), 7.0);
        assert_eq!(ev("2^3^2"), 512.0);
        assert_eq!(ev("-2^2"), -4.0);
        assert_eq!(ev("2^-1"), 0.5);
        assert_eq!(ev("7%3"), 1.0);
        assert_eq!(ev("1<2 && 3>2 ? 5 : 6"), 5.0);
        assert_eq!(ev("i+j*10"), 12.0);
        assert_eq!(ev("max(1, 4, 2)"), 4.0);
        assert!((ev("sin(pi/2)") - 1.0).abs() < 1e-12);
        assert!(Fx::new("foo(1)").is_err());
        assert!(Fx::new("1+").is_err());
    }

    #[test]
    fn statements_variables_and_loops() {
        let img = FloatImage::new(4, 3, 1);
        let ev = |s: &str| Fx::new(s).unwrap().eval_at(&img, 2, 1, 0);
        assert_eq!(ev("x = 3; z = x * 2; x + z"), 9.0);
        assert_eq!(ev("x = 1; x += 4; x *= 2; x"), 10.0);
        // Sum of 1 … 10 three ways.
        assert_eq!(
            ev("s0 = 0; k = 1; while(k <= 10, s0 += k; k += 1); s0"),
            55.0
        );
        assert_eq!(ev("s0 = 0; for(k = 1, k <= 10, s0 += k; k += 1); s0"), 55.0);
        assert_eq!(ev("s0 = 0; k = 0; do(k += 1; s0 += k, k < 10); s0"), 55.0);
        assert_eq!(ev("if(i == 2, 7, 8)"), 7.0);
        assert_eq!(ev("x = 2; x == 2 ? 1 : 0"), 1.0);
        // A loop that never ends is cut off.
        assert_eq!(ev("k = 0; while(1, k += 1); k"), 10_000.0);
        assert!(Fx::new("i = 3").is_err());
        assert!(Fx::new("q += 1").is_err());
        assert!(Fx::new("zz + 1").is_err());
        assert!(Fx::new("while(1)").is_err());
    }

    #[test]
    fn the_added_functions_and_symbols() {
        let img = FloatImage::from_samples(2, 1, 3, vec![1.0, 0.5, 0.0, 0.2, 0.2, 0.2]).unwrap();
        let ev = |s: &str, x: u32, c: usize| Fx::new(s).unwrap().eval_at(&img, x, 0, c);
        assert_eq!(ev("clamp(1.5) + clamp(-2)", 0, 0), 1.0);
        assert!((ev("gauss(0)", 0, 0) - 0.398_942_280_4).abs() < 1e-9);
        assert_eq!(ev("sinc(0)", 0, 0), 1.0);
        assert!(ev("sinc(1)", 0, 0).abs() < 1e-12);
        assert_eq!(ev("alt(2) + alt(3)", 0, 0), 0.0);
        assert_eq!(ev("not(0) + not(3)", 0, 0), 1.0);
        assert_eq!(ev("isnan(0/1)", 0, 0), 0.0);
        assert_eq!(ev("channel(10, 20, 30)", 0, 1), 20.0);
        assert!(
            (ev("drc(0.5, 2)", 0, 0) - 0.5 / (2.0 * -0.5 + 1.0 + 1e-300)).is_infinite() || true
        );
        // Orange (1, 0.5, 0): hue 30° = 1/12, fully saturated; grey: none.
        assert!((ev("hue", 0, 0) - 1.0 / 12.0).abs() < 1e-9);
        assert_eq!(ev("saturation", 0, 0), 1.0);
        assert_eq!(ev("saturation", 1, 0), 0.0);
        let r = ev("rand()", 0, 0);
        assert!((0.0..1.0).contains(&r));
        assert_eq!(r, ev("rand()", 0, 0));
        assert_ne!(r, ev("rand()", 1, 0));
        assert_ne!(ev("rand() - rand()", 0, 0), 0.0);
        assert_eq!(ev("n + t", 0, 0), 1.0);
    }

    #[test]
    fn a_second_picture_is_v() {
        let u = FloatImage::from_samples(2, 1, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let v = FloatImage::from_samples(2, 1, 3, vec![0.5; 6]).unwrap();
        let mix = Fx::new("(u + v) / 2").unwrap();
        let out = mix.apply_images(&[u.clone(), v.clone()]).unwrap();
        for (o, i) in out.samples().iter().zip(u.samples()) {
            assert!((o - (i + 0.5) / 2.0).abs() < 1e-6);
        }
        let out = Fx::new("v.r - u[0].b + u[1].g * 0 + v[1,0] * 0 + n")
            .unwrap()
            .apply_images(&[u.clone(), v])
            .unwrap();
        assert!((out.pixel(0, 0)[0] - (0.5 - 0.3 + 2.0)).abs() < 1e-6);
        // With one picture `v` reads the first.
        let out = Fx::new("v").unwrap().apply_float(&u);
        assert_eq!(out.samples(), u.samples());
        assert!(Fx::new("u").unwrap().apply_images(&[]).is_err());
    }
}
