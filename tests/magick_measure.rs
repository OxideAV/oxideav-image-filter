//! Statistics, compare metrics, moments, histogram and bounds against
//! `magick` (ImageMagick 7.1.2 Q16-HDRI), live when the binary is on
//! the path: `identify -verbose` channel statistics, `compare -metric`
//! on `rose:` versus its `-blur 0x1`, `identify -moments`, `%k` and
//! `%@`. Without `magick` the test reports itself skipped.

use std::process::Command;

use oxideav_core::{PixelFormat, VideoFrame, VideoPlane};
use oxideav_image_filter::VideoStreamParams;
use oxideav_image_filter::measure::{
    CompareMetric, compare, moments, statistics, trim_bounds, unique_colors,
};

/// `true` for the reference build (ImageMagick 7.1 Q16-HDRI; other builds clamp or round differently).
fn is_reference_build(version: &[u8]) -> bool {
    let v = String::from_utf8_lossy(version);
    v.contains("ImageMagick 7.1") && v.contains("Q16-HDRI")
}

fn magick() -> bool {
    Command::new("magick")
        .arg("-version")
        .output()
        .map(|o| o.status.success() && is_reference_build(&o.stdout))
        .unwrap_or(false)
}

fn run(args: &[&str]) -> Vec<u8> {
    let out = Command::new("magick").args(args).output().unwrap();
    assert!(
        out.status.success(),
        "magick {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn run_text(args: &[&str]) -> String {
    let out = Command::new("magick").args(args).output().unwrap();
    // `compare` reports the metric on stderr and exits 1 when the images differ.
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

fn frame(rgb: Vec<u8>, w: u32) -> VideoFrame {
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w as usize * 3,
            data: rgb,
        }],
    }
}

const P: VideoStreamParams = VideoStreamParams {
    format: PixelFormat::Rgb24,
    width: 70,
    height: 46,
};

/// The number after `label:` in an `identify -verbose` block, reading
/// the normalised figure in parentheses when present.
fn field(block: &str, label: &str) -> f64 {
    let line = block
        .lines()
        .find(|l| l.trim_start().starts_with(label))
        .unwrap_or_else(|| panic!("{label} in {block}"));
    let rest = line.split(':').nth(1).unwrap().trim();
    if let Some(open) = rest.find('(') {
        let close = rest.find(')').unwrap();
        rest[open + 1..close].parse().unwrap()
    } else {
        rest.split(',').next().unwrap().trim().parse().unwrap()
    }
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: got {got}, magick says {want}"
    );
}

#[test]
fn rose_statistics_compare_moments_histogram_bounds_match_magick() {
    if !magick() {
        eprintln!("reference magick not on the path: live measurement comparison skipped");
        return;
    }
    let rose = frame(run(&["rose:", "-depth", "8", "rgb:-"]), 70);
    let blur = frame(run(&["rose:", "-blur", "0x1", "-depth", "8", "rgb:-"]), 70);

    // --- channel statistics
    let verbose = run_text(&["rose:", "-depth", "8", "-verbose", "info:"]);
    let stats = statistics(&rose, P).unwrap();
    let stats_at = verbose.find("Channel statistics:").unwrap();
    for (ci, name) in ["Red:", "Green:", "Blue:"].iter().enumerate() {
        let start = verbose[stats_at..].find(name).unwrap() + stats_at;
        let block = &verbose[start..start + 400];
        let s = stats.channels[ci];
        close(s.min, field(block, "min:"), 1e-6, &format!("{name} min"));
        close(s.max, field(block, "max:"), 1e-6, &format!("{name} max"));
        close(s.mean, field(block, "mean:"), 1e-5, &format!("{name} mean"));
        close(
            s.median,
            field(block, "median:"),
            1e-5,
            &format!("{name} median"),
        );
        close(
            s.std_dev,
            field(block, "standard deviation:"),
            1e-4,
            &format!("{name} std"),
        );
        close(
            s.kurtosis,
            field(block, "kurtosis:"),
            1e-3,
            &format!("{name} kurtosis"),
        );
        close(
            s.skewness,
            field(block, "skewness:"),
            1e-3,
            &format!("{name} skewness"),
        );
        close(
            s.entropy,
            field(block, "entropy:"),
            1e-4,
            &format!("{name} entropy"),
        );
    }
    let start = verbose[stats_at..].find("Overall:").unwrap() + stats_at;
    let block = &verbose[start..start + 400];
    close(
        stats.overall.mean,
        field(block, "mean:"),
        1e-5,
        "overall mean",
    );
    close(
        stats.overall.std_dev,
        field(block, "standard deviation:"),
        1e-4,
        "overall std",
    );
    close(
        stats.overall.entropy,
        field(block, "entropy:"),
        1e-4,
        "overall entropy",
    );

    // --- compare metrics (the parenthesised normalised figure)
    let rose_png = std::env::temp_dir().join("oxideav_measure_rose.png");
    let blur_png = std::env::temp_dir().join("oxideav_measure_rose_blur.png");
    run(&["rose:", rose_png.to_str().unwrap()]);
    run(&["rose:", "-blur", "0x1", blur_png.to_str().unwrap()]);
    for (metric, tol) in [
        (CompareMetric::MeanAbsoluteError, 1e-4),
        (CompareMetric::MeanSquaredError, 1e-4),
        (CompareMetric::RootMeanSquaredError, 1e-4),
        (CompareMetric::PeakAbsoluteError, 1e-6),
        (CompareMetric::PeakSignalToNoiseRatio, 1e-3),
        (CompareMetric::NormalizedCrossCorrelation, 1e-3),
        (CompareMetric::StructuralSimilarity, 2e-3),
        (CompareMetric::StructuralDissimilarity, 2e-3),
    ] {
        let text = run_text(&[
            "compare",
            "-metric",
            metric.name(),
            rose_png.to_str().unwrap(),
            blur_png.to_str().unwrap(),
            "null:",
        ]);
        let want: f64 = if metric == CompareMetric::PeakSignalToNoiseRatio {
            text.split_whitespace().next().unwrap().parse().unwrap()
        } else {
            let open = text.find('(').unwrap();
            let close_ = text.find(')').unwrap();
            text[open + 1..close_].parse().unwrap()
        };
        let got = compare(&rose, &blur, P, metric, 0.0).unwrap().value;
        close(got, want, tol, metric.name());
    }
    // MEPP's third figure is the normalised peak.
    let text = run_text(&[
        "compare",
        "-metric",
        "MEPP",
        rose_png.to_str().unwrap(),
        blur_png.to_str().unwrap(),
        "null:",
    ]);
    let inner = &text[text.find('(').unwrap() + 1..text.find(')').unwrap()];
    let mut parts = inner.split(',').map(|v| v.trim().parse::<f64>().unwrap());
    let (want_mean, want_peak) = (parts.next().unwrap(), parts.next().unwrap());
    let mepp = compare(&rose, &blur, P, CompareMetric::MeanErrorPerPixel, 0.0).unwrap();
    close(mepp.value, want_mean, 1e-4, "MEPP mean");
    close(mepp.peak, want_peak, 1e-6, "MEPP peak");

    // --- moments (red channel + overall)
    let mom = run_text(&[
        "identify",
        "-verbose",
        "-moments",
        rose_png.to_str().unwrap(),
    ]);
    let (per, overall) = moments(&rose, P).unwrap();
    for (block_name, m) in [("Red:", per[0]), ("Overall:", overall)] {
        let hu_start = mom.find("Channel Hu moments:").unwrap();
        let start = mom[hu_start..].find(block_name).unwrap() + hu_start;
        let block = &mom[start..start + 700];
        let centroid = block.lines().find(|l| l.contains("Centroid:")).unwrap();
        let c: Vec<f64> = centroid
            .split(':')
            .nth(1)
            .unwrap()
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        close(
            m.centroid.0,
            c[0],
            1e-4,
            &format!("{block_name} centroid x"),
        );
        close(
            m.centroid.1,
            c[1],
            1e-4,
            &format!("{block_name} centroid y"),
        );
        let axes = block
            .lines()
            .find(|l| l.contains("Semi-Major/Minor"))
            .unwrap();
        let a: Vec<f64> = axes
            .split(':')
            .nth(1)
            .unwrap()
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        close(
            m.semi_major,
            a[0],
            1e-4,
            &format!("{block_name} semi-major"),
        );
        close(
            m.semi_minor,
            a[1],
            1e-4,
            &format!("{block_name} semi-minor"),
        );
        close(
            m.angle_degrees,
            field(block, "Ellipse angle:"),
            1e-4,
            &format!("{block_name} angle"),
        );
        close(
            m.eccentricity,
            field(block, "Ellipse eccentricity:"),
            1e-4,
            &format!("{block_name} eccentricity"),
        );
        close(
            m.intensity,
            field(block, "Ellipse intensity:"),
            1e-4,
            &format!("{block_name} intensity"),
        );
        for i in 0..8 {
            let want = field(block, &format!("I{}:", i + 1));
            assert!(
                (m.hu[i] - want).abs() <= 1e-3 * want.abs().max(1e-6),
                "{block_name} I{}: got {} magick {want}",
                i + 1,
                m.hu[i]
            );
        }
    }

    // --- unique colours and trim bounds
    let k: usize = run_text(&["identify", "-format", "%k", rose_png.to_str().unwrap()])
        .trim()
        .parse()
        .unwrap();
    assert_eq!(unique_colors(&rose, P).unwrap(), k);
    let bordered = frame(
        run(&[
            "rose:",
            "-bordercolor",
            "black",
            "-border",
            "5",
            "-depth",
            "8",
            "rgb:-",
        ]),
        80,
    );
    let bp = VideoStreamParams {
        format: PixelFormat::Rgb24,
        width: 80,
        height: 56,
    };
    let want = run_text(&[
        "rose:",
        "-bordercolor",
        "black",
        "-border",
        "5",
        "-format",
        "%@",
        "info:",
    ]);
    assert_eq!(want.trim(), "70x46+5+5");
    assert_eq!(
        trim_bounds(&bordered, bp, 0.0, None).unwrap(),
        Some((5, 5, 70, 46))
    );
    let _ = std::fs::remove_file(rose_png);
    let _ = std::fs::remove_file(blur_png);
}
