//! SVG loading and rasterization via ThorVG.
//!
//! SVGs are vector graphics, so — unlike raster formats — there is no single
//! "full resolution" to decode. To keep them crisp at *any* zoom level we render
//! the SVG to a pixel buffer at the viewer's current displayed size (intrinsic ×
//! zoom), re-rendering only when that size drifts past a threshold (pan never
//! re-renders; it only shifts the same texture). The result feeds the same
//! `texture` draw path used by every raster image, so the existing
//! fit/zoom/cursor-anchor/pan/viewport-clamp code is reused unchanged: the SVG
//! is treated as a logical image whose pixel dimensions are its intrinsic size.
//!
//! Re-renders are capped at [`SVG_RENDER_CAP`] to bound memory and time; beyond
//! that the GPU upscales the cap-sized raster (still sharp well past typical use).

use egui::ColorImage;
use std::cell::RefCell;
use thorvg::{ColorSpace, EngineOption, MimeType, Matrix, Paint, Thorvg};
#[cfg(test)]
use thorvg::{Rect, Rgba};

/// Safety cap on the longest side of an SVG render, for the rare case of a
/// viewport larger than the GPU's max texture side (e.g. huge HiDPI monitor).
/// Under normal display sizes this never bites; the viewport-crop render is
/// always viewport-sized, so memory tracks the screen, not the zoom.
pub const SVG_RENDER_CAP: u32 = 8192;

/// Parse the SVG's intrinsic pixel extent from its `width`/`height` attributes,
/// falling back to the `viewBox` aspect, then to 512×512. This is the "virtual
/// pixel" size of the SVG: `zoom == 1` means one SVG user unit maps to one
/// screen pixel, and its aspect drives `is_scaled_to_fit` exactly like a raster.
pub fn svg_intrinsic_size(bytes: &[u8]) -> [f32; 2] {
    let s = std::str::from_utf8(bytes).unwrap_or("");
    let width = find_attr_value(s, "width").and_then(parse_number);
    let height = find_attr_value(s, "height").and_then(parse_number);
    if let (Some(w), Some(h)) = (width, height) {
        if w > 0.0 && h > 0.0 {
            return [w, h];
        }
    }
    if let Some(vb) = find_attr_value(s, "viewBox").and_then(viewbox_size) {
        return vb;
    }
    [512.0, 512.0]
}

/// Render `data` (SVG bytes) into a `canvas_w` × `canvas_h` RGBA [`ColorImage`],
/// applying an affine transform `m = scale(s) · translate(tx, ty)` that maps SVG
/// user-space into the canvas. This is a **viewport-crop** render: the caller
/// passes a viewport-sized canvas plus the transform that lands exactly the
/// on-screen region into `[0, canvas_w) × [0, canvas_h)`. Because the texture is
/// always viewport-sized (device pixels) and the draw covers exactly the
/// viewport, it is sample-for-sample 1:1 with the screen at **any** zoom level
/// — including absurd zoom-in, where the visible region is a tiny crop of a
/// conceptually huge vector image. The texture is never upscaled (the only
/// thing that blurs); memory is bounded by the screen, never the zoom.
///
/// The canvas starts fully transparent, so letterbox areas (where no SVG
/// content lands) composite to the viewer's panel background, exactly like
/// raster images with transparency.
///
/// ThorVG writes [`ColorSpace::ABGR8888`] — a `u32` with channels A,B,G,R MSB
/// to LSB — which on little-endian hosts lays out in memory as `R,G,B,A` bytes
/// per pixel, exactly what [`ColorImage::from_rgba_unmultiplied`] expects.
pub fn render_svg_viewport(
    data: &[u8],
    canvas_w: u32,
    canvas_h: u32,
    s: f32,
    tx: f32,
    ty: f32,
) -> Result<ColorImage, String> {
    if canvas_w == 0 || canvas_h == 0 {
        return Err(format!("refusing zero-size SVG render ({canvas_w}x{canvas_h})"));
    }
    with_engine(|eng| {
        let mut buf = vec![0u32; (canvas_w as usize) * (canvas_h as usize)];

        let mut canvas = eng
            .sw_canvas(EngineOption::Default)
            .map_err(|e| format!("sw_canvas: {e:?}"))?;
        // SAFETY: set_target writes into `buf` with stride == canvas_w and the
        // given dimensions; we allocated exactly canvas_w * canvas_h u32s.
        unsafe {
            canvas
                .set_target(&mut buf, canvas_w, canvas_w, canvas_h, ColorSpace::ABGR8888)
                .map_err(|e| format!("set_target: {e:?}"))?;
        }

        let m = Matrix::IDENTITY.scale(s, s).translate(tx, ty);
        let mut pic = eng.picture().map_err(|e| format!("picture: {e:?}"))?;
        pic.load_data(data, MimeType::Svg, None)
            .map_err(|e| format!("load_data: {e:?}"))?;
        pic.set_transform(&m)
            .map_err(|e| format!("set_transform: {e:?}"))?;
        canvas.add(pic).map_err(|e| format!("add(pic): {e:?}"))?;

        canvas.draw(true).map_err(|e| format!("draw: {e:?}"))?;
        canvas.sync().map_err(|e| format!("sync: {e:?}"))?;

        // Reinterpret the ABGR8888 u32 buffer as RGBA8 bytes (no shuffle:
        // little-endian ABGR lays out as R,G,B,A in memory).
        let rgba: Vec<u8> = buf.into_iter().flat_map(|px| px.to_le_bytes()).collect();
        Ok(ColorImage::from_rgba_unmultiplied(
            [canvas_w as usize, canvas_h as usize],
            &rgba,
        ))
    })
}

// (Background-only helper kept for the byte-order regression test below.)
#[cfg(test)]
fn render_svg_solid_bg(data: &[u8], w: u32, h: u32) -> Result<ColorImage, String> {
    if w == 0 || h == 0 {
        return Err(format!("refusing zero-size SVG render ({w}x{h})"));
    }
    with_engine(|eng| {
        let mut buf = vec![0u32; (w as usize) * (h as usize)];
        let mut canvas = eng
            .sw_canvas(EngineOption::Default)
            .map_err(|e| format!("sw_canvas: {e:?}"))?;
        unsafe {
            canvas
                .set_target(&mut buf, w, w, h, ColorSpace::ABGR8888)
                .map_err(|e| format!("set_target: {e:?}"))?;
        }
        let mut bg = eng.shape().map_err(|e| format!("shape: {e:?}"))?;
        bg.append_rect(Rect::new(0.0, 0.0, w as f32, h as f32))
            .map_err(|e| format!("append_rect: {e:?}"))?;
        bg.set_fill_color(Rgba::new(255, 255, 255, 255))
            .map_err(|e| format!("set_fill_color: {e:?}"))?;
        canvas.add(bg).map_err(|e| format!("add(bg): {e:?}"))?;
        let m = Matrix::IDENTITY.scale(w as f32, h as f32).translate(0.0, 0.0);
        let mut pic = eng.picture().map_err(|e| format!("picture: {e:?}"))?;
        pic.load_data(data, MimeType::Svg, None)
            .map_err(|e| format!("load_data: {e:?}"))?;
        pic.set_transform(&m)
            .map_err(|e| format!("set_transform: {e:?}"))?;
        canvas.add(pic).map_err(|e| format!("add(pic): {e:?}"))?;
        canvas.draw(true).map_err(|e| format!("draw: {e:?}"))?;
        canvas.sync().map_err(|e| format!("sync: {e:?}"))?;
        let rgba: Vec<u8> = buf.into_iter().flat_map(|px| px.to_le_bytes()).collect();
        Ok(ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba))
    })
}

// ──────────────────────────────────────────────────────────────────────────
// ThorVG engine is a process-global resource. We lazily init it once per thread
// on first use and reuse it for every subsequent render. (Renders happen on the
// UI thread only — SVG decoding hands the *bytes* to the UI thread, which then
// rasterizes at the current zoom; ThorVG is never touched off-thread.)
thread_local! {
    static TVG: RefCell<Option<Thorvg>> = const { RefCell::new(None) };
}

fn with_engine<R>(f: impl FnOnce(&Thorvg) -> R) -> R {
    TVG.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(
                Thorvg::init(0).expect("ThorVG::init failed; cannot render SVG"),
            );
        }
        f(slot.as_ref().expect("engine just initialized"))
    })
}

// ── tiny attribute parser (avoids pulling regex as a direct dependency) ──

/// Find the value of an XML attribute `attr="..."` (or `'...'`) on the root
/// `<svg>` tag, scanning from the start of the document. Word-boundary aware so
/// `width` does not match inside `stroke-width`.
fn find_attr_value<'a>(s: &'a str, attr: &str) -> Option<&'a str> {
    let bytes = s.as_bytes();
    let pat = attr.as_bytes();
    let mut i = 0;
    while i + pat.len() <= bytes.len() {
        if &bytes[i..i + pat.len()] == pat {
            let before = if i == 0 { b' ' } else { bytes[i - 1] };
            if before.is_ascii_alphanumeric() || before == b'_' || before == b'-' {
                i += 1;
                continue;
            }
            let mut j = i + pat.len();
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= bytes.len() || bytes[j] != b'=' {
                i += pat.len();
                continue;
            }
            j += 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j >= bytes.len() {
                return None;
            }
            let q = bytes[j];
            if q == b'"' || q == b'\'' {
                let start = j + 1;
                let end = s[start..].find(q as char)? + start;
                return Some(&s[start..end]);
            }
            // Unquoted attribute value (rare in SVG): read until whitespace/>.
            let start = j;
            let end = bytes[start..]
                .iter()
                .position(|c| c.is_ascii_whitespace() || *c == b'>' || *c == b'/')
                .map(|p| start + p)
                .unwrap_or(bytes.len());
            return Some(&s[start..end]);
        }
        i += 1;
    }
    None
}

/// Parse a leading number from an attribute value, treating `N`, `Npx`, `Npt`
/// (pt≈px for our purposes) as absolute pixel values. Returns `None` for
/// relative units (`%`, `em`, `ex`) which cannot give a fixed pixel extent.
fn parse_number(v: &str) -> Option<f32> {
    let v = v.trim();
    if v.contains('%') || v.contains("em") {
        return None;
    }
    let num: String = v
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+'))
        .collect();
    let n: f32 = num.parse().ok()?;
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    Some(n)
}

/// Extract `(width, height)` from a `viewBox="minx miny w h"` value.
fn viewbox_size(v: &str) -> Option<[f32; 2]> {
    let nums: Vec<f32> = v
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let (w, h) = match nums.len() {
        n if n >= 4 => (nums[2], nums[3]),
        n if n >= 2 => (nums[n - 2], nums[n - 1]),
        _ => return None,
    };
    (w > 0.0 && h > 0.0).then_some([w, h])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsic_from_width_height() {
        let s = br#"<svg xmlns="x" width="640" height="480"><rect/></svg>"#;
        assert_eq!(svg_intrinsic_size(s), [640.0, 480.0]);
    }

    #[test]
    fn intrinsic_falls_back_to_viewbox() {
        let s = br#"<svg xmlns="x" viewBox="0 0 300 200"><rect/></svg>"#;
        assert_eq!(svg_intrinsic_size(s), [300.0, 200.0]);
    }

    #[test]
    fn intrinsic_stroke_width_does_not_match_width() {
        // `width` must not be matched inside `stroke-width` (a real bug source).
        let s = br#"<svg xmlns="x" viewBox="0 0 100 50"><line stroke-width="9"/></svg>"#;
        assert_eq!(svg_intrinsic_size(s), [100.0, 50.0]);
    }

    #[test]
    fn intrinsic_default_for_unsized() {
        let s = br#"<svg xmlns="x"><circle r="10"/></svg>"#;
        assert_eq!(svg_intrinsic_size(s), [512.0, 512.0]);
    }

    #[test]
    fn render_red_fill_is_red_pixel_abgr_to_rgba() {
        // Whole-canvas red fill that fills the viewBox → the decoded pixel must be
        // pure red. This verifies the ABGR8888→RGBA8 byte-ordering used by render_svg:
        // a swapped mapping would come out blue.
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 4 4">
            <rect x="0" y="0" width="4" height="4" fill="rgb(255,0,0)"/>
        </svg>"#;
        let img = render_svg_solid_bg(svg, 4, 4).expect("render");
        assert_eq!(img.size, [4, 4]);
        let px = img.pixels[0]; // egui Color32: r,g,b,a
        assert_eq!(px.r(), 255, "red channel");
        assert!(px.g() < 16, "green near zero, got {}", px.g());
        assert!(px.b() < 16, "blue near zero, got {}", px.b());
        assert_eq!(px.a(), 255, "opaque white background → fully opaque");
    }
}
