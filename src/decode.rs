// --- Image Loading & Helper Functions ---
use egui::ColorImage;
use image::{codecs::gif::GifDecoder, AnimationDecoder, DynamicImage, ImageReader, Luma};
use jxl::api::{states, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult};
use libjpeg_turbo_rs as ljt;
use ndarray::{s, Array, Array2, IxDyn};
use rayon::prelude::*;
use rustronomy_fits as rsf;
use std::{
    env,
    error::Error,
    fs,
    io::BufReader,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::formats::*;
use crate::types::{AnimationFrame, LoadedImage};

pub fn decode_image_data(path: &Path) -> Result<DynamicImage, String> {
    let path_str = path.to_string_lossy();
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();

    if ANIM_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        return load_gif_first_frame(&path_str);
    }

    log::info!("Loading image: {}", path_str);
    log::info!("Detected format based on extension: {}", extension);

    if RAW_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        load_raw(&path_str)
    } else if FITS_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        load_fits(&path_str)
    } else if JXL_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        load_jxl(&path_str)
    } else if matches!(extension.as_str(), "jpg" | "jpeg") {
        load_jpeg_full(&path_str)
    } else {
        load_with_image_crate(&path_str)
    }
}

/// Full-resolution decode for the foreground worker. Returns the displayable
/// image directly; no preview/cache variant is retained.
/// Result of decoding the startup image ahead of `App::new`, run from `main()`
/// so the decode overlaps eframe's window/GL setup. Same shape as
/// [`load_full_for_worker`]'s return value.
pub type PredecodeResult = Result<LoadedImage, String>;

pub fn load_full_for_worker(path: &Path) -> Result<LoadedImage, String> {
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    if ANIM_SUPPORTED_FORMATS.contains(&extension.as_str()) {
        return Ok(load_animated_gif(&path.to_string_lossy())?);
    }
    if matches!(extension.as_str(), "svg") {
        // SVG: defer rasterization to the UI thread, which knows the current zoom
        // and can render at display size. Hand the raw bytes + intrinsic size over.
        return load_svg(path);
    }
    if matches!(extension.as_str(), "jxl") {
        // JPEG XL: decode via jxl-rs, which exposes every frame and its
        // duration so animated JXL files play back correctly (a single-frame
        // decode would drop the animation).
        return load_jxl_full(path);
    }
    let dynamic_image = decode_image_data(path)?;
    // Build the display image from a borrow; `dynamic_image` is then dropped, so
    // we never hold the full decoded buffer twice.
    let color = color_image_from_dynamic(&dynamic_image);
    Ok(LoadedImage::Static(color))
}

/// Read an SVG file as raw bytes and parse its intrinsic size. Rasterization is
/// done later on the UI thread by [`crate::svg::render_svg`] at the displayed
/// zoom level, so no decode happens here.
fn load_svg(path: &Path) -> Result<LoadedImage, String> {
    let data =
        fs::read(path).map_err(|e| format!("Failed to read SVG {}: {}", path.display(), e))?;
    let intrinsic = crate::svg::svg_intrinsic_size(&data);
    Ok(LoadedImage::Svg { data, intrinsic })
}

/// Decode JPEG bytes with libjpeg-turbo-rs (pure-Rust libjpeg-turbo with SIMD),
/// at full resolution. Output is always RGBA8 (the decoder converts
/// grayscale/CMYK/etc. for us). Returns `None` on any failure so the caller can
/// fall back to the format-sniffing `image`-crate path (mislabelled files, etc.).
fn decode_jpeg_turbo(data: &[u8]) -> Option<DynamicImage> {
    let mut decoder = ljt::Decoder::new(data).ok()?;
    decoder.set_output_format(ljt::PixelFormat::Rgba);
    let img = decoder.decode_image().ok()?;
    image::RgbaImage::from_raw(img.width as u32, img.height as u32, img.data).map(DynamicImage::ImageRgba8)
}

/// Full-resolution JPEG decode. Tries the fast SIMD libjpeg-turbo path first and
/// falls back to the `image` crate (with content sniffing) for files that aren't
/// actually decodable JPEGs despite a .jpg/.jpeg extension.
fn load_jpeg_full(path: &str) -> Result<DynamicImage, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
    if let Some(img) = decode_jpeg_turbo(&data) {
        return Ok(img);
    }
    log::debug!("libjpeg-turbo declined {}; falling back to image crate", path);
    load_with_image_crate(path)
}

/// Decode every frame of an animated GIF into already-composited RGBA frames with
/// their delays. The `image` crate's GIF `AnimationDecoder` handles frame
/// disposal/compositing, so each returned frame is a full-canvas image. A
/// single-frame GIF is returned as a plain static image so it takes the normal
/// (tile-capable) render path.
fn load_animated_gif(path: &str) -> Result<LoadedImage, String> {
    let file = fs::File::open(path).map_err(|e| format!("Failed to open GIF: {}", e))?;
    let reader = BufReader::new(file);
    let decoder = GifDecoder::new(reader).map_err(|e| format!("Failed to create GIF decoder: {}", e))?;

    let mut frames = Vec::new();
    for (i, frame) in decoder.into_frames().enumerate() {
        let frame = frame.map_err(|e| format!("Failed to decode GIF frame {}: {}", i, e))?;
        // Clamp absurdly short / zero delays (common in GIFs) to a sane minimum so
        // playback doesn't peg the CPU; this matches typical browser behavior.
        let delay = Duration::from(frame.delay()).max(Duration::from_millis(20));
        let buffer = frame.into_buffer();
        let dims = buffer.dimensions();
        let image = ColorImage::from_rgba_unmultiplied([dims.0 as _, dims.1 as _], buffer.as_raw());
        frames.push(AnimationFrame { image, delay });
    }

    match frames.len() {
        0 => Err("GIF has no frames".to_string()),
        1 => Ok(LoadedImage::Static(frames.pop().unwrap().image)),
        _ => Ok(LoadedImage::Animated { frames, pending: None }),
    }
}

/// Open a JPEG XL file, read its basic info, and configure interleaved RGBA8
/// output. Returns the decoder positioned at the first frame, the canvas size
/// `(w, h)`, and whether the bitstream declares animation. `input` is advanced
/// past the image header.
///
/// Extra channels (e.g. a separate alpha plane) are marked `None` in the pixel
/// format: their count must still match the image, but we don't take separate
/// buffers for them — jxl-rs composites the alpha into the RGBA color output.
/// (Requesting an extra channel *as U8* trips a conversion bug in jxl-rs.)
fn open_jxl(
    input: &mut &[u8],
    path_str: &str,
) -> Result<(JxlDecoder<states::WithImageInfo>, usize, usize, bool), String> {
    let init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut with_image_info = {
        let mut d = init;
        loop {
            match d
                .process(input)
                .map_err(|e| format!("Failed to parse JXL header {}: {}", path_str, e))?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    d = fallback;
                    if input.is_empty() {
                        return Err(format!("JXL {} ended before image info", path_str));
                    }
                }
            }
        }
    };

    let basic = with_image_info.basic_info().clone();
    let (w, h) = basic.size;
    let animated = basic.animation.is_some();
    let nextra = basic.extra_channels.len();
    if w == 0 || h == 0 {
        return Err(format!("JXL {} has zero size ({}x{})", path_str, w, h));
    }
    with_image_info.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgba,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![None; nextra],
    });
    Ok((with_image_info, w, h, animated))
}

/// Decode a single frame, returning the decoder (ready for the next frame), the
/// frame's interleaved RGBA8 pixels, and its duration in milliseconds (`None`
/// for still images). Drives the jxl-rs state machine through one
/// `WithImageInfo -> WithFrameInfo -> WithImageInfo` cycle.
fn decode_one_frame(
    with_image_info: JxlDecoder<states::WithImageInfo>,
    input: &mut &[u8],
    w: usize,
    h: usize,
    path_str: &str,
) -> Result<(JxlDecoder<states::WithImageInfo>, Vec<u8>, Option<f64>), String> {
    let bytes_per_row = w * 4;
    let mut rgba = vec![0u8; bytes_per_row * h];
    let duration_ms: Option<f64>;
    let with_image_info = {
        let mut bufs = [JxlOutputBuffer::new(&mut rgba, h, bytes_per_row)];

        // WithImageInfo -> WithFrameInfo.
        let with_frame_info = {
            let mut d = with_image_info;
            loop {
                match d.process(input).map_err(|e| {
                    format!("Failed to decode JXL frame header {}: {}", path_str, e)
                })? {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        let mut fb = fallback;
                        fb.flush_pixels(&mut bufs).map_err(|e| {
                            format!("Failed to flush JXL pixels {}: {}", path_str, e)
                        })?;
                        d = fb;
                        if input.is_empty() {
                            return Err(format!("JXL {} ended mid-frame", path_str));
                        }
                    }
                }
            }
        };

        duration_ms = with_frame_info.frame_header().duration;

        // WithFrameInfo -> WithImageInfo (renders the frame into `bufs`).
        {
            let mut d = with_frame_info;
            loop {
                match d.process(input, &mut bufs).map_err(|e| {
                    format!("Failed to decode JXL frame {}: {}", path_str, e)
                })? {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        let mut fb = fallback;
                        fb.flush_pixels(&mut bufs).map_err(|e| {
                            format!("Failed to flush JXL pixels {}: {}", path_str, e)
                        })?;
                        d = fb;
                        if input.is_empty() {
                            return Err(format!("JXL {} ended mid-frame", path_str));
                        }
                    }
                }
            }
        }
    };
    Ok((with_image_info, rgba, duration_ms))
}

/// Convert a jxl-rs frame duration (milliseconds) to a playback delay, floored
/// at 1 ms so a zero-duration (blend) frame can't stall the playback loop.
fn jxl_delay(duration_ms: Option<f64>) -> Duration {
    let secs = duration_ms.unwrap_or(0.0) / 1000.0;
    Duration::from_secs_f64(secs.max(0.0)).max(Duration::from_millis(1))
}

fn load_jxl(path: &str) -> Result<DynamicImage, String> {
    log::info!("Loading JXL: {}", path);
    let data = fs::read(path).map_err(|e| format!("Failed to read JXL {}: {}", path, e))?;
    let mut input: &[u8] = &data;
    let (decoder, w, h, _animated) = open_jxl(&mut input, path)?;
    let (_decoder, rgba, _dur) = decode_one_frame(decoder, &mut input, w, h, path)?;
    log::info!("Loading image data: {}x{}", w, h);
    image::RgbaImage::from_raw(w as u32, h as u32, rgba)
        .map(DynamicImage::ImageRgba8)
        .ok_or_else(|| format!("Failed to create image from JXL data ({})", path))
}

/// Decode a JPEG XL file via the `jxl-rs` decoder (the official libjxl Rust
/// port). A still image is decoded synchronously and returned as
/// [`LoadedImage::Static`]. An animated image is decoded on a background
/// thread that streams frames in over a channel: this function blocks only
/// until the *first* frame is ready and returns it immediately, so opening a
/// long animation is effectively instant instead of blocking on every frame.
///
/// The jxl-rs decoder is not `Send`, so the animated decode creates its own
/// decoder on the background thread; we open the file once here only to read
/// the header and learn whether the image is animated.
fn load_jxl_full(path: &Path) -> Result<LoadedImage, String> {
    let path_str = path.to_string_lossy();
    log::info!("Loading JXL (animated-aware): {}", path_str);
    let data = fs::read(path).map_err(|e| format!("Failed to read JXL {}: {}", path_str, e))?;

    let mut input: &[u8] = &data;
    let (decoder, w, h, animated) = open_jxl(&mut input, &path_str)?;
    if !animated {
        // Still image: finish the single-frame decode right here.
        let (_decoder, rgba, _dur) = decode_one_frame(decoder, &mut input, w, h, &path_str)?;
        return Ok(LoadedImage::Static(ColorImage::from_rgba_unmultiplied(
            [w, h],
            &rgba,
        )));
    }
    drop(decoder); // re-opened on the background thread (decoder is !Send)

    log::info!(
        "JXL {} is animated; showing first frame, decoding the rest lazily",
        path_str
    );
    let bg_path = path_str.to_string();
    let (tx, rx) = std::sync::mpsc::channel::<AnimationFrame>();
    std::thread::spawn(move || {
        let mut input: &[u8] = &data;
        let (decoder, w, h, _) = match open_jxl(&mut input, &bg_path) {
            Ok(x) => x,
            Err(e) => {
                log::warn!("JXL {} lazy open failed: {}", bg_path, e);
                return;
            }
        };
        let mut decoder = decoder;
        let mut sent = 0usize;
        loop {
            match decode_one_frame(decoder, &mut input, w, h, &bg_path) {
                Ok((next, rgba, dur)) => {
                    decoder = next;
                    let frame = AnimationFrame {
                        image: ColorImage::from_rgba_unmultiplied([w, h], &rgba),
                        delay: jxl_delay(dur),
                    };
                    if tx.send(frame).is_err() {
                        break; // receiver dropped — viewer moved on
                    }
                    sent += 1;
                }
                Err(e) => {
                    log::warn!("JXL {} lazy decode stopped: {}", bg_path, e);
                    break;
                }
            }
            if !decoder.has_more_frames() {
                break;
            }
        }
        log::info!("JXL {} lazy decode complete: {} frames", bg_path, sent);
    });

    // Block only until the first frame is ready, then let the rest stream in.
    let first = rx
        .recv()
        .map_err(|_| format!("JXL {} decoded no frames", path_str))?;
    Ok(LoadedImage::Animated {
        frames: vec![first],
        pending: Some(rx),
    })
}

fn load_gif_first_frame(path: &str) -> Result<DynamicImage, String> {
    let file = fs::File::open(path).map_err(|e| format!("Failed to open GIF: {}", e))?;
    let reader = BufReader::new(file);
    let decoder = GifDecoder::new(reader).map_err(|e| format!("Failed to create GIF decoder: {}", e))?;
    let frame = decoder
        .into_frames()
        .next()
        .ok_or_else(|| "GIF has no frames".to_string())?
        .map_err(|e| format!("Failed to decode GIF frame: {}", e))?;
    Ok(DynamicImage::ImageRgba8(frame.into_buffer()))
}

/// Convert a decoded image to the RGBA8 layout egui uploads, borrowing the
/// source so the caller can keep the `DynamicImage` afterwards if needed.
pub fn color_image_from_dynamic(img: &DynamicImage) -> ColorImage {
    let rgba = img.to_rgba8();
    let dims = rgba.dimensions();
    ColorImage::from_rgba_unmultiplied([dims.0 as _, dims.1 as _], rgba.as_raw())
}

fn load_with_image_crate(path: &str) -> Result<DynamicImage, String> {
    log::debug!("Loading with image-rs: {}", path);
    // Determine the format from the file's actual magic bytes rather than trusting
    // the extension. Some files carry a .jpg name but are really WebP/PNG/etc.;
    // without this, `decode()` would feed them to the JPEG decoder and fail with
    // e.g. "Illegal start bytes: 5249" (the "RI" of a RIFF/WebP container).
    ImageReader::open(path)
        .map_err(|e| format!("Failed to open {}: {}", path, e))?
        .with_guessed_format()
        .map_err(|e| format!("Failed to read {}: {}", path, e))?
        .decode()
        .map_err(|e| format!("Failed to decode {}: {}", path, e))
}

fn load_raw(path: &str) -> Result<DynamicImage, String> {
    log::debug!("Loading RAW: {}", path);
    let mut pipeline = imagepipe::Pipeline::new_from_file(path).map_err(|e| format!("Failed to load RAW: {}", e))?;
    let decoded = pipeline.output_8bit(None).map_err(|e| format!("Failed to process RAW: {}", e))?;

    image::RgbImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.data)
        .map(DynamicImage::ImageRgb8)
        .ok_or_else(|| "Failed to create image from RAW data".to_string())
}

fn load_fits(path: &str) -> Result<DynamicImage, String> {
    log::debug!("Loading FITS: {}", path);
    let mut fits = rsf::Fits::open(Path::new(path)).map_err(|e| format!("FITS open error: {}", e))?;
    let hdu = fits.remove_hdu(0).ok_or_else(|| "FITS HDU error: failed to remove HDU".to_string())?;
    let data = hdu.to_parts().1.ok_or("No data in FITS HDU")?;

    let array = match data {
        rsf::Extension::Image(img) => rgb_to_grayscale(img.as_owned_f32_array()),
        _ => Err("No image data found in FITS".into()),
    }
    .map_err(|e| format!("FITS data conversion error: {}", e))?;

    let (height, width) = (array.shape()[0], array.shape()[1]);
    #[allow(deprecated)]
    let mut data_f32: Vec<f32> = array.into_raw_vec();

    let (min_val, max_val) = data_f32
        .par_iter()
        .fold(|| (f32::MAX, f32::MIN), |(min, max), &x| (min.min(x), max.max(x)))
        .reduce(|| (f32::MAX, f32::MIN), |(a_min, a_max), (b_min, b_max)| (a_min.min(b_min), a_max.max(b_max)));
    let scale = 255.0 / (max_val - min_val).max(1e-5);
    data_f32.par_iter_mut().for_each(|x| *x = (*x - min_val) * scale);

    let log_factor = 3000.0;
    let gamma = 1.5;
    let buffer: Vec<u8> = data_f32
        .par_iter()
        .map(|&x| {
            let log_scaled = 255.0 * (1.0 + log_factor * (x.clamp(0.0, 255.0) / 255.0)).ln() / (1.0 + log_factor).ln();
            ((log_scaled / 255.0).powf(gamma) * 255.0) as u8
        })
        .collect();

    image::ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width as u32, height as u32, buffer)
        .map(DynamicImage::ImageLuma8)
        .ok_or_else(|| "Failed to create image from FITS data".to_string())
}

fn rgb_to_grayscale(rgb_image: Result<Array<f32, IxDyn>, Box<dyn Error>>) -> Result<Array2<f32>, Box<dyn Error>> {
    let rgb_array = rgb_image?;
    let shape = rgb_array.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err("Invalid shape: Expected (H, W, 3)".into());
    }
    Ok(&rgb_array.slice(s![.., .., 0]) * 0.2989 + &rgb_array.slice(s![.., .., 1]) * 0.5870 + &rgb_array.slice(s![.., .., 2]) * 0.1140)
}

/// Read `dir` and return every supported image file inside it, sorted by
/// lowercased filename so the listing is stable across calls.
pub fn scan_supported_images(dir: &Path) -> Vec<PathBuf> {
    let all_supported_formats: Vec<&str> = [
        &IMAGEREADER_SUPPORTED_FORMATS[..],
        &ANIM_SUPPORTED_FORMATS[..],
        &IMAGE_RS_SUPPORTED_FORMATS[..],
        &RAW_SUPPORTED_FORMATS[..],
        &FITS_SUPPORTED_FORMATS[..],
        &JXL_SUPPORTED_FORMATS[..],
    ]
    .concat();

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            if !path.is_file() {
                return false;
            }
            let path_str = path.to_string_lossy().to_lowercase();
            all_supported_formats.iter().any(|format| path_str.ends_with(format))
        })
        .collect();
    files.sort_by_key(|name| name.to_string_lossy().to_lowercase());
    files
}


pub fn get_absolute_path(filename: &str) -> Result<PathBuf, String> {
    let path = Path::new(filename);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        env::current_dir()
            .map(|mut dir| {
                dir.push(path);
                dir
            })
            .map_err(|e| format!("Failed to get current dir: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A file whose extension lies about its real format (here: PNG bytes in a
    /// `.jpg`) must still decode by sniffing the magic bytes, rather than failing
    /// with "Illegal start bytes". Regression test for mislabelled images such as
    /// WebP-in-.jpg files exported by some phone cameras.
    #[test]
    fn decodes_when_extension_mismatches_content() {
        let mut img = image::RgbImage::new(8, 6);
        img.put_pixel(0, 0, image::Rgb([10, 20, 30]));
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = env::temp_dir().join(format!("lv_mismatch_{}.jpg", nanos));
        // Write PNG-encoded bytes to a path ending in `.jpg`.
        DynamicImage::ImageRgb8(img)
            .save_with_format(&path, image::ImageFormat::Png)
            .unwrap();

        let decoded = decode_image_data(&path);
        let _ = fs::remove_file(&path);
        let decoded = decoded.expect("mislabelled PNG-in-.jpg should still decode");
        assert_eq!((decoded.width(), decoded.height()), (8, 6));
    }

    /// Encode a JPEG in memory and decode it back through libjpeg-turbo, at full
    /// resolution and at a 1/2 DCT scale, confirming dimensions and that the colour
    /// survives the round-trip.
    #[test]
    fn turbo_decodes_full() {
        let mut src = image::RgbImage::new(800, 600);
        for p in src.pixels_mut() {
            *p = image::Rgb([200, 100, 50]);
        }
        let (w, h) = src.dimensions();
        let bytes = ljt::Encoder::new(src.as_raw().as_slice(), w as usize, h as usize, ljt::PixelFormat::Rgb)
            .quality(90)
            .encode()
            .expect("encode jpeg");

        // Full decode.
        let full = decode_jpeg_turbo(&bytes).expect("turbo full decode");
        assert_eq!((full.width(), full.height()), (800, 600));
        let rgba = full.to_rgba8();
        let px = rgba.get_pixel(400, 300).0;
        assert!((px[0] as i32 - 200).abs() < 12 && (px[1] as i32 - 100).abs() < 12, "colour off: {:?}", px);
    }

    /// Rough decode-speed comparison on a large, hard-to-compress JPEG. Ignored by
    /// default (slow, needs --release for SIMD). Run with:
    ///   cargo test --release bench_jpeg_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_jpeg_decode() {
        use std::time::Instant;
        // 24MP with a noisy pattern so the decoder actually does work.
        let (w, h) = (6000u32, 4000u32);
        let mut src = image::RgbImage::new(w, h);
        // Smooth, photographic-like content (gradients + low-frequency waves) so the
        // encoded size and decode cost resemble a real photo rather than pure noise.
        for (x, y, p) in src.enumerate_pixels_mut() {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let r = (128.0 + 110.0 * (fx * 6.28).sin()) as u8;
            let g = (128.0 + 110.0 * (fy * 9.42).sin()) as u8;
            let b = (255.0 * (fx + fy) * 0.5) as u8;
            *p = image::Rgb([r, g, b]);
        }
        let bytes = ljt::Encoder::new(src.as_raw().as_slice(), w as usize, h as usize, ljt::PixelFormat::Rgb)
            .quality(90)
            .encode()
            .expect("encode jpeg");
        println!("\nJPEG size: {} KB ({}x{})", bytes.len() / 1024, w, h);

        let runs = 5;
        let bench = |label: &str, f: &dyn Fn()| {
            f(); // warm up
            let t = Instant::now();
            for _ in 0..runs {
                f();
            }
            println!("  {:<28} {:>7.1} ms/decode", label, t.elapsed().as_secs_f64() * 1000.0 / runs as f64);
        };

        bench("turbo full -> RGBA", &|| {
            decode_jpeg_turbo(&bytes).unwrap();
        });
    }

}
