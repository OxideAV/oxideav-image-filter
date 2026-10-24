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
//! Integer output saturates to the code range; float output keeps the
//! raw result.

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
        absolute: bool,
        x: Box<Expr>,
        y: Box<Expr>,
        channel: Option<usize>,
    },
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
}

/// A parsed `-fx` expression.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Fx {
    /// The source text.
    pub expression: String,
    ast: Expr,
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
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
                let e = self.ternary()?;
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
                    let x = self.ternary()?;
                    self.expect(",")?;
                    let y = self.ternary()?;
                    self.expect(if absolute { "}" } else { "]" })?;
                    let channel = self.channel_suffix()?;
                    return Ok(Expr::Pixel {
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
                            args.push(self.ternary()?);
                            if self.eat(")") {
                                break;
                            }
                            self.expect(",")?;
                        }
                    }
                    return Ok(Expr::Call(name.to_ascii_lowercase(), args));
                }
                if name == "u" || name == "p" || name == "s" {
                    if let Some(c) = self.channel_suffix()? {
                        return Ok(Expr::Var(Var::Channel(c)));
                    }
                    return Ok(Expr::Var(Var::U));
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
                    other => return Err(perr(format!("unknown symbol '{other}'"))),
                })
            }
            Some(c) => Err(perr(format!("unexpected '{}' at {}", c as char, self.pos))),
        }
    }
}

struct Ctx<'a> {
    img: &'a FloatImage,
    x: u32,
    y: u32,
    c: usize,
}

impl Ctx<'_> {
    fn channel(&self, x: f64, y: f64, c: usize) -> f64 {
        let ch = self.img.channels();
        let c = if ch == 1 { 0 } else { c.min(ch - 1) };
        if c == 3 && ch < 4 {
            return 1.0;
        }
        let (fx, fy) = (x.floor(), y.floor());
        let (tx, ty) = (x - fx, y - fy);
        let s = |xx: f64, yy: f64| self.img.sample_clamped(xx as i64, yy as i64, c) as f64;
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
            Var::Intensity => {
                let g = |c| cx.channel(cx.x as f64, cx.y as f64, c);
                if cx.img.channels() < 3 {
                    g(0)
                } else {
                    0.212_656 * g(0) + 0.715_158 * g(1) + 0.072_186 * g(2)
                }
            }
        },
        Expr::Pixel {
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
            cx.channel(px, py, channel.unwrap_or(cx.c))
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
                _ => f64::NAN,
            }
        }
    }
}

const FUNCTIONS: &[&str] = &[
    "abs", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "exp", "floor", "hypot", "int",
    "trunc", "ln", "log", "logtwo", "max", "min", "mod", "pow", "round", "sign", "sin", "sinh",
    "sqrt", "squish", "tan", "tanh",
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
        _ => Ok(()),
    }
}

impl Fx {
    /// Parse an expression.
    pub fn new(expression: &str) -> Result<Self, Error> {
        let mut p = Parser {
            s: expression.as_bytes(),
            pos: 0,
        };
        let ast = p.ternary()?;
        p.ws();
        if p.pos != p.s.len() {
            return Err(perr(format!("trailing input at {}", p.pos)));
        }
        check_calls(&ast)?;
        Ok(Self {
            expression: expression.to_string(),
            ast,
        })
    }

    /// Evaluate at pixel `(x, y)` for channel `c` of `img`.
    pub fn eval_at(&self, img: &FloatImage, x: u32, y: u32, c: usize) -> f64 {
        eval(&self.ast, &Ctx { img, x, y, c })
    }

    /// Apply to every colour channel of a float picture (alpha kept).
    pub fn apply_float(&self, img: &FloatImage) -> FloatImage {
        let mut out = img.clone();
        let cc = img.color_channels();
        for y in 0..img.height() {
            for x in 0..img.width() {
                for c in 0..cc {
                    out.pixel_mut(x, y)[c] = self.eval_at(img, x, y, c) as f32;
                }
            }
        }
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
}
