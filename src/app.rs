/// Process-start anchor; set as the very first statement of `main()` so every
/// startup milestone is measured against true process birth.
pub static PROC_START: OnceLock<Instant> = OnceLock::new();

/// Milliseconds since process start (for `[prof]` timing markers).
pub fn proc_ms() -> f64 {
    PROC_START.get().map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0)
}

// One-shot startup paint milestone (static so the paint branch — which already
// holds `&mut self.image` — can update it without borrowing `self`).
static PROF_FIRST_PAINT: AtomicBool = AtomicBool::new(false);

fn prof_paint_milestone(tex_w: usize, tex_h: usize) {
    use std::sync::atomic::Ordering;
    if !PROF_FIRST_PAINT.swap(true, Ordering::SeqCst) {
        eprintln!(
            "[prof] T+{:.1}ms >>> FIRST PAINT — full-res texture on screen ({}x{})",
            proc_ms(), tex_w, tex_h
        );
    }
}

/// Scale a per-frame drag delta componentwise so the image can't be pushed past a
/// viewport wall: for a component that would move the image *outside* the valid
/// offset range, dampen toward the wall instead of overshooting. The component
/// sliding *along* a free axis (or already in range, i.e. mid-image pan) is kept
/// unchanged so the drag stays responsive everywhere except at the walls.
///
/// Because the delta itself is constrained (rather than the accumulated offset),
/// there is no out-of-bounds offset building up during a drag — so a later hard
/// clamp never has to snap an overshoot back, and the cursor never "slips" off the
/// pixel the user is dragging toward. Zoom is left untouched: this only governs
/// manual panning.
fn dampen_drag_delta(delta: Vec2, offset: Vec2, scaled_image_size: Vec2, view_dim: Vec2) -> Vec2 {
    let dampen_axis = |d: f32, off: f32, img: f32, view: f32| -> f32 {
        if img >= view {
            // Image covers / exceeds the viewport: keep it covering — valid offset is
            // [view - img, 0]. Zero motion that would expose more than one edge.
            let lo = view - img;
            let hi = 0.0;
            let next = off + d;
            if d < 0.0 && next < lo {
                (lo - off).max(d) // ease into the lower wall, never overshoot
            } else if d > 0.0 && next > hi {
                (hi - off).min(d) // ease into the upper wall
            } else {
                d
            }
        } else {
            // Image smaller than the viewport: keep it fully inside — valid offset is
            // [0, view - img]. Same easing against the walls.
            let lo = 0.0;
            let hi = view - img;
            let next = off + d;
            if d < 0.0 && next < lo {
                (lo - off).max(d)
            } else if d > 0.0 && next > hi {
                (hi - off).min(d)
            } else {
                d
            }
        }
    };
    Vec2::new(
        dampen_axis(delta.x, offset.x, scaled_image_size.x, view_dim.x),
        dampen_axis(delta.y, offset.y, scaled_image_size.y, view_dim.y),
    )
}
use eframe::egui;
use egui::{epaint::RectShape, Color32, ColorImage, Pos2, Rect, Shape, Vec2};
use eframe::glow::HasContext;
use arboard::{Clipboard, ImageData};
use std::{
    borrow::Cow,
    sync::OnceLock,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

use crate::decode::scan_supported_images;
use crate::types::{
    Animation, AnimationFrame, DisplayableImage, FullResRequest, FullResReply, FullResWorker,
    LoadedImage, MemoryGate, SvgRenderState, SvgSource,
};
use crate::workers::spawn_full_res_worker;

const TILE_SIZE: usize = 1024; // Use tiles of 1024x1024 pixels for the detail view
/// Maximum time we wait for a full-res decode before assuming the worker is stuck
/// (slow/hung decoder, bad file). After this we respawn the worker and unblock
/// the bulk preload so the app doesn't sit there silently forever.
const FULL_RES_WATCHDOG: Duration = Duration::from_secs(20);

pub struct ImageViewerApp {
    image: Option<DisplayableImage>,
    image_files: Vec<PathBuf>,
    current_index: usize,
    image_order: Vec<usize>,
    zoom: f32,
    offset: Vec2,
    velocity: Vec2,
    is_scaled_to_fit: bool,
    /// Has the user taken manual control of the view (panned or zoomed) for the
    /// current image? While false and `is_scaled_to_fit` is also false, the view
    /// auto-fits at "actual size or zoomed-out-to-fit" (zoom capped at 1.0, so no
    /// image is ever zoomed in on open). Any pan/scroll-zoom flips this true and
    /// pins the current zoom/offset until the next image is loaded.
    interacted: bool,
    /// GPU maximum 2D texture side, queried once from the glow (OpenGL) context
    /// at startup via GL_MAX_TEXTURE_SIZE. Images larger than this in either
    /// dimension are tiled; smaller ones are uploaded whole (no resize).
    max_texture_side: usize,
    is_fullscreen: bool,
    is_randomized: bool,
    show_delete_confirmation: bool,
    last_error: Option<String>,
    clipboard: Option<Clipboard>,
    full_res_pending: bool,
    full_res_pending_since: Option<Instant>,
    full_res_worker: Option<FullResWorker>,
    memory_gate: Arc<MemoryGate>,
    /// Pre-decode of the startup image, spawned in `main()` before window setup
    /// so its decode overlaps eframe's window/GL creation. Consumed (or polled)
    /// in `load_image_at_index` / `check_pending_load`.
    predecode: Option<(PathBuf, std::sync::mpsc::Receiver<crate::decode::PredecodeResult>)>,
}

impl ImageViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        path: Option<PathBuf>,
        initial_fullscreen: bool,
        predecode: Option<(PathBuf, std::sync::mpsc::Receiver<crate::decode::PredecodeResult>)>,
    ) -> Self {
        let memory_gate = Arc::new(MemoryGate::new());
        let full_res_worker = Some(spawn_full_res_worker(cc.egui_ctx.clone(), memory_gate.clone()));
        // Query the real GPU 2D-texture limit instead of a hardcoded guess. On the
        // glow (OpenGL) backend cc.gl is the live GL context, already current.
        let max_texture_side = cc
            .gl
            .as_ref()
            .and_then(|gl| {
                let v = unsafe { gl.get_parameter_i32(eframe::glow::MAX_TEXTURE_SIZE) };
                if v > 0 { Some(v as usize) } else { None }
            })
            .unwrap_or(16384); // sane modern fallback if the context is somehow absent
        let mut app = Self {
            image: None,
            image_files: Vec::new(),
            current_index: 0,
            image_order: Vec::new(),
            zoom: 1.0,
            offset: Vec2::ZERO,
            velocity: Vec2::ZERO,
            is_scaled_to_fit: false,
            interacted: false,
            max_texture_side,
            is_fullscreen: initial_fullscreen,
            is_randomized: false,
            show_delete_confirmation: false,
            last_error: None,
            clipboard: Clipboard::new().ok(),
            full_res_pending: false,
            full_res_pending_since: None,
            full_res_worker,
            memory_gate,
            predecode,
        };
        if let Some(path) = path {
            eprintln!("[prof] T+{:.1}ms App::new entry, loading '{}'", proc_ms(), path.display());
            let t_gather = Instant::now();
            app.gather_images_from_directory(&path);
            eprintln!(
                "[prof] T+{:.1}ms directory scanned: {} images (scan took {:.1}ms)",
                proc_ms(), app.image_files.len(), t_gather.elapsed().as_secs_f64() * 1000.0
            );
            if !app.image_files.is_empty() {
                app.load_image_at_index(app.current_index, &cc.egui_ctx);
            } else {
                app.last_error = Some(format!("No supported images found in directory of '{}'", path.display()));
            }
        } else {
            app.last_error = Some("No image file specified.".to_string());
        }
        app
    }

    fn load_image_at_index(&mut self, index: usize, ctx: &egui::Context) {
        self.current_index = index;
        let path = self.image_files[self.image_order[self.current_index]].clone();
        log::info!("Loading image: {}", path.display());
        self.is_scaled_to_fit = false;
        self.interacted = false;
        self.velocity = Vec2::ZERO;
        self.full_res_pending = false;
        self.full_res_pending_since = None;

        // Route the decode through the worker so the UI thread stays responsive
        // and the user can keep navigating; the central panel renders a
        // “Loading…” placeholder until the full-res reply arrives. (We deliberately
        // skip the embedded EXIF thumbnail: it is usually a 4:3 crop whose aspect
        // rarely matches the full image, producing a brief “stretched a few
        // pixels” flash on load.)
        eprintln!("[prof] T+{:.1}ms load_image_at_index: full-res worker path", proc_ms());
        self.image = None;
        self.last_error = None;
        // Prefer an in-flight pre-decode (spawned in main before window setup)
        // over spinning a fresh worker — the decode should already be done.
        if let Some((pd_path, rx)) = self.predecode.take() {
            if pd_path == path {
                match rx.try_recv() {
                    Ok(res) => {
                        eprintln!(
                            "[prof] T+{:.1}ms predecode: ready at App::new (decode overlapped window setup)",
                            proc_ms()
                        );
                        self.consume_predecode(pd_path, res, ctx);
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        eprintln!(
                            "[prof] T+{:.1}ms predecode: still running; will poll in check_pending_load",
                            proc_ms()
                        );
                        self.predecode = Some((pd_path, rx));
                        self.full_res_pending = true;
                        self.full_res_pending_since = Some(Instant::now());
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        log::warn!("Pre-decode thread died for {}; falling back to worker.", pd_path.display());
                        self.start_full_res_load(path, ctx);
                    }
                }
            } else {
                // predecode is for a different path (navigated before first
                // paint); drop it and use the normal worker path.
                self.start_full_res_load(path, ctx);
            }
        } else {
            self.start_full_res_load(path, ctx);
        }
        ctx.request_repaint();
    }

    fn display_loaded_image(&mut self, image: ColorImage, path: &Path, ctx: &egui::Context) {
        let max_texture_side = self.max_texture_side;
        let needs_tiling = image.width() > max_texture_side || image.height() > max_texture_side;

        // Tiled images (larger than the GPU's max texture side) render directly from
        // full-res tiles and get no single texture; everything else uploads the
        // whole image as a single GPU texture.
        let texture = if needs_tiling {
            None
        } else {
            Some(ctx.load_texture(
                path.display().to_string(),
                image.clone(),
                Default::default(),
            ))
        };

        eprintln!(
            "[prof] T+{:.1}ms texture ready src {}x{} (needs_tiling={})",
            proc_ms(), image.width(), image.height(),
            needs_tiling
        );

        self.image = Some(DisplayableImage {
            full_res_image: image,
            texture,
            tile_cache: HashMap::new(),
            needs_tiling,
            animation: None,
            svg: None,
        });

        self.last_error = None;
    }

    /// Install an animated image for playback. Animated frames always render via
    /// the simple non-tiled path: each tick the UI swaps `texture` for the active
    /// frame, so we never tile them. The first frame is shown immediately and
    /// `full_res_image` tracks the displayed frame (used for sizing and clipboard
    /// copies). `pending` (when `Some`) streams the not-yet-decoded frames in from
    /// a background decoder; the playback loop drains it as frames arrive.
    fn display_animated_image(
        &mut self,
        frames: Vec<AnimationFrame>,
        pending: Option<std::sync::mpsc::Receiver<AnimationFrame>>,
        path: &Path,
        ctx: &egui::Context,
    ) {
        let first_frame = frames[0].image.clone();
        let texture = Some(ctx.load_texture(
            format!("{}_anim", path.display()),
            first_frame.clone(),
            Default::default(),
        ));

        self.image = Some(DisplayableImage {
            full_res_image: first_frame,
            texture,
            tile_cache: HashMap::new(),
            needs_tiling: false,
            animation: Some(Animation {
                frames,
                current: 0,
                frame_started: Instant::now(),
                pending,
            }),
            svg: None,
        });

        self.last_error = None;
    }

    /// Install an SVG image for display. No rasterization happens here: we store
    /// the bytes + intrinsic size and leave `texture` as `None`; the draw
    /// loop rasterizes on the first frame (and re-rasterizes whenever the zoom
    /// drifts) at the current displayed resolution for crisp-at-any-zoom output.
    /// `full_res_image` starts as a 1×1 placeholder and is updated to the latest
    /// rendered raster on each re-render so clipboard copy reflects what's shown.
    fn display_svg_image(
        &mut self,
        data: Vec<u8>,
        intrinsic: [f32; 2],
        path: &Path,
        ctx: &egui::Context,
    ) {
        let _ = (ctx, path); // kept for symmetry with display_loaded_image / logging
        eprintln!(
            "[prof] T+{:.1}ms texture ready [svg] intrinsic {}x{} (deferred rasterize)",
            proc_ms(),
            intrinsic[0] as u32,
            intrinsic[1] as u32
        );
        self.image = Some(DisplayableImage {
            full_res_image: ColorImage::from_rgba_unmultiplied([1, 1], &[0, 0, 0, 0]),
            texture: None,
            tile_cache: HashMap::new(),
            needs_tiling: false,
            animation: None,
            svg: Some(SvgSource { data, intrinsic, render_state: None }),
        });
        self.last_error = None;
    }

    fn start_full_res_load(&mut self, path: PathBuf, ctx: &egui::Context) {
        let request = FullResRequest { path: path.clone() };
        let send_result = self
            .full_res_worker
            .as_ref()
            .map(|w| w.tx.send(request));
        // If the worker channel is gone (e.g. it panicked out of catch_unwind), respawn it
        // so subsequent navigations still get full-res loads.
        if !matches!(send_result, Some(Ok(()))) {
            log::warn!("Full-res worker unavailable; respawning.");
            let worker = spawn_full_res_worker(ctx.clone(), self.memory_gate.clone());
            let _ = worker.tx.send(FullResRequest { path: path.clone() });
            self.full_res_worker = Some(worker);
        }
        self.full_res_pending = true;
        self.full_res_pending_since = Some(Instant::now());
        // Ensure the watchdog gets a chance to run even if the UI stays idle.
        ctx.request_repaint_after(FULL_RES_WATCHDOG);
    }

    fn check_pending_load(&mut self, ctx: &egui::Context) {
        // Watchdog: if a full-res decode hasn't returned for too long, the worker is
        // likely stuck on a slow/bad file. Drop it so the next nav respawns a fresh one.
        if self.full_res_pending {
            if let Some(since) = self.full_res_pending_since {
                if since.elapsed() > FULL_RES_WATCHDOG {
                    log::warn!(
                        "Full-res worker stuck for {:.1?}; respawning on next navigation.",
                        since.elapsed()
                    );
                    self.full_res_worker = None;
                    self.full_res_pending = false;
                    self.full_res_pending_since = None;
                }
            }
        }

        // Pre-decode (started in `main` before eframe window setup): poll it first so
        // the parallel decode result is displayed without waiting on the worker.
        if let Some((pd_path, rx)) = self.predecode.take() {
            let current_path = self
                .image_files
                .get(self.image_order.get(self.current_index).copied().unwrap_or(usize::MAX))
                .cloned();
            if current_path.as_ref() == Some(&pd_path) {
                match rx.try_recv() {
                    Ok(res) => {
                        self.consume_predecode(pd_path, res, ctx);
                        return;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        // Still decoding; keep the slot, ensure the watchdog tracks it,
                        // and re-poll on a tight cadence (sub-frame) so the result
                        // lands as soon as the decode finishes.
                        self.predecode = Some((pd_path, rx));
                        self.full_res_pending = true;
                        if self.full_res_pending_since.is_none() {
                            self.full_res_pending_since = Some(Instant::now());
                        }
                        ctx.request_repaint_after(Duration::from_millis(4));
                        return;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        log::warn!("Pre-decode thread died for {}; falling back to worker.", pd_path.display());
                        self.start_full_res_load(pd_path, ctx);
                        return;
                    }
                }
            } else {
                log::debug!("Dropping stale pre-decode for {}", pd_path.display());
            }
        }

        let Some(worker) = self.full_res_worker.as_ref() else { return };
        // Drain all available replies, skipping stale ones (path doesn't match current).
        loop {
            let reply = match worker.rx.try_recv() {
                Ok(r) => r,
                Err(std::sync::mpsc::TryRecvError::Empty) => return,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Worker died; reset so the next nav respawns it.
                    self.full_res_worker = None;
                    self.full_res_pending = false;
                    self.full_res_pending_since = None;
                    return;
                }
            };

            let current_path = self
                .image_files
                .get(self.image_order.get(self.current_index).copied().unwrap_or(usize::MAX))
                .cloned();
            if current_path.as_ref() != Some(&reply.path) {
                log::debug!("Discarding stale full-res reply for {}", reply.path.display());
                continue;
            }

            self.apply_full_res_reply(reply, ctx);
            return;
        }
    }

    /// Apply a full-res worker reply: clear pending state, swap the image in, and
    /// Apply a full-res worker reply: clear pending state and swap the image in
    /// (the only kind of reply now, since no preview is ever produced).
    /// Extracted so the pre-decode path reuses the exact same display logic as the
    /// worker path.
    fn apply_full_res_reply(&mut self, reply: FullResReply, ctx: &egui::Context) {
        self.full_res_pending = false;
        self.full_res_pending_since = None;
        match reply.result {
            Ok(loaded) => {
                match loaded {
                    LoadedImage::Static(full_res) => {
                        self.display_loaded_image(full_res, &reply.path, ctx)
                    },
                    LoadedImage::Animated { frames, pending } => {
                        self.display_animated_image(frames, pending, &reply.path, ctx)
                    }
                    LoadedImage::Svg { data, intrinsic } => {
                        self.display_svg_image(data, intrinsic, &reply.path, ctx)
                    },
                }
                eprintln!("[prof] T+{:.1}ms check_pending_load: FULL-RES reply swapped in", proc_ms());
                log::info!("Swapped in full-res image: {}", reply.path.display());
                ctx.request_repaint();
            }
            Err(e) => {
                log::error!("Background full-res load failed for {}: {}", reply.path.display(), e);
                if self.image.is_none() {
                    self.last_error = Some(e);
                }
            }
        }
    }

    /// Consume a result from the startup pre-decode thread (parallel with window
    /// setup). Mirrors how the worker handles the same decode: route through
    /// `apply_full_res_reply` for identical display behavior.
    fn consume_predecode(
        &mut self,
        path: PathBuf,
        res: crate::decode::PredecodeResult,
        ctx: &egui::Context,
    ) {
        let reply = match res {
            Ok(loaded) => {
                eprintln!(
                    "[prof] T+{:.1}ms predecode: consumed (decode overlapped window setup)",
                    proc_ms()
                );
                FullResReply { path, result: Ok(loaded) }
            }
            Err(e) => {
                log::error!("Pre-decode failed for {}: {}", path.display(), e);
                FullResReply { path, result: Err(e) }
            }
        };
        self.apply_full_res_reply(reply, ctx);
    }

    fn shutdown_workers(&mut self) {
        // Dropping the worker drops the request Sender, which causes the worker
        // thread's recv() to fail and exit.
        self.full_res_worker = None;
    }

    fn copy_to_clipboard(&mut self) {
        if let (Some(clipboard), Some(image)) = (&mut self.clipboard, &self.image) {
            let image = &image.full_res_image;

            let rgba_bytes: Vec<u8> = image
                .pixels
                .iter()
                .flat_map(|color| color.to_array())
                .collect();

            let image_data = ImageData {
                width: image.width(),
                height: image.height(),
                bytes: Cow::from(rgba_bytes),
            };
            log::info!("Copying image: {}x{}", image_data.width, image_data.height);
            if let Err(e) = clipboard.set_image(image_data) {
                self.last_error = Some(format!("Failed to copy to clipboard: {}", e));
            } else {
                log::info!("Image copied to clipboard.");
            }
        }
    }

    fn gather_images_from_directory(&mut self, file_path: &Path) {
        let parent_dir = match file_path.parent() {
            Some(p) => p,
            None => {
                self.last_error = Some("Failed to get parent directory.".to_string());
                return;
            }
        };

        let files = scan_supported_images(parent_dir);

        if let Some(index) = files.iter().position(|p| p == file_path) {
            self.current_index = index;
        }

        self.image_files = files;
        self.image_order = (0..self.image_files.len()).collect();
    }

    /// Re-scan the parent directory so files added/removed externally show up the
    /// next time the user navigates. Keeps the currently-viewed image in place,
    /// preserves random-order traversal, and only restarts bulk preload if the
    /// listing actually changed (so this is cheap to call on every navigation).
    fn refresh_directory(&mut self) {
        let parent_dir = match self
            .image_files
            .get(self.image_order.get(self.current_index).copied().unwrap_or(usize::MAX))
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .or_else(|| self.image_files.first().and_then(|p| p.parent()).map(|p| p.to_path_buf()))
        {
            Some(p) => p,
            None => return,
        };

        let new_files = scan_supported_images(&parent_dir);
        if new_files == self.image_files {
            return;
        }

        log::info!(
            "Directory contents changed: {} -> {} files",
            self.image_files.len(),
            new_files.len()
        );

        let current_path = self
            .image_files
            .get(self.image_order.get(self.current_index).copied().unwrap_or(usize::MAX))
            .cloned();

        if self.is_randomized {
            // Preserve the user's random traversal order: walk the old order, drop
            // entries whose path no longer exists, then append any new files.
            let old_path_order: Vec<PathBuf> = self
                .image_order
                .iter()
                .filter_map(|&i| self.image_files.get(i).cloned())
                .collect();
            let mut new_order = Vec::with_capacity(new_files.len());
            let mut seen = vec![false; new_files.len()];
            for path in &old_path_order {
                if let Some(idx) = new_files.iter().position(|p| p == path) {
                    if !seen[idx] {
                        seen[idx] = true;
                        new_order.push(idx);
                    }
                }
            }
            for (idx, was_seen) in seen.iter().enumerate() {
                if !was_seen {
                    new_order.push(idx);
                }
            }
            self.image_order = new_order;
        } else {
            self.image_order = (0..new_files.len()).collect();
        }

        self.image_files = new_files;

        // Restore current_index to point at the same path. If that file was deleted
        // externally we clamp so the next nav step lands on a valid entry.
        if let Some(cp) = current_path {
            if let Some(file_idx) = self.image_files.iter().position(|p| p == &cp) {
                if let Some(order_idx) = self.image_order.iter().position(|&i| i == file_idx) {
                    self.current_index = order_idx;
                }
            } else if self.current_index >= self.image_order.len() {
                self.current_index = self.image_order.len().saturating_sub(1);
            }
        }
    }

    fn next_image(&mut self, ctx: &egui::Context) {
        self.refresh_directory();
        if !self.image_files.is_empty() {
            self.load_image_at_index((self.current_index + 1) % self.image_files.len(), ctx);
        }
    }

    fn prev_image(&mut self, ctx: &egui::Context) {
        self.refresh_directory();
        if !self.image_files.is_empty() {
            self.load_image_at_index((self.current_index + self.image_files.len() - 1) % self.image_files.len(), ctx);
        }
    }

    fn first_image(&mut self, ctx: &egui::Context) {
        self.refresh_directory();
        if !self.image_files.is_empty() {
            self.load_image_at_index(0, ctx);
        }
    }

    fn last_image(&mut self, ctx: &egui::Context) {
        self.refresh_directory();
        if !self.image_files.is_empty() {
            self.load_image_at_index(self.image_files.len() - 1, ctx);
        }
    }

    fn handle_keyboard_input(&mut self, ctx: &egui::Context) {

        let events = ctx.input(|i| i.events.clone());
        // Iterate over all events that occurred this frame.
        for event in &events {
            // Pattern match to find the `Copy` event.
            if let egui::Event::Copy = event {
                log::info!("Copying image to clipboard...");
                self.copy_to_clipboard();
            }
        }

        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            self.next_image(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            self.prev_image(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Home)) {
            self.first_image(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::End)) {
            self.last_image(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.show_delete_confirmation {
                self.show_delete_confirmation = false;
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Q)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            self.is_fullscreen = !self.is_fullscreen;
        }
        if !self.show_delete_confirmation && ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
            self.is_scaled_to_fit = !self.is_scaled_to_fit;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            self.show_delete_confirmation = true;
        }
    }
}

impl eframe::App for ImageViewerApp {

fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
    let ctx = ui.ctx().clone();
    let is_currently_fullscreen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
    if self.is_fullscreen != is_currently_fullscreen {
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.is_fullscreen));
    }

    self.handle_keyboard_input(&ctx);
    self.check_pending_load(&ctx);

    egui::CentralPanel::default()
        .frame(egui::Frame::default().fill(Color32::from_rgb(20, 20, 20)))
        .show_inside(ui, |ui| {
            if let Some(image) = &mut self.image {
                // Advance animation playback (GIFs). Frames are pre-decoded; we step
                // forward by wall-clock time and re-upload the active frame whenever it
                // changes, then schedule a repaint for when the next frame is due.
                if let Some(anim) = &mut image.animation {
                    // Drain any lazily-decoded frames that have arrived from the
                    // background decoder (JXL streams frames in after the first).
                    if let Some(rx) = &anim.pending {
                        use std::sync::mpsc::TryRecvError;
                        loop {
                            match rx.try_recv() {
                                Ok(frame) => anim.frames.push(frame),
                                Err(TryRecvError::Empty) => break,
                                Err(TryRecvError::Disconnected) => {
                                    anim.pending = None;
                                    break;
                                }
                            }
                        }
                    }

                    if anim.frames.len() > 1 {
                        let mut changed = false;
                        // Advance past every frame whose display time has elapsed. The
                        // step cap guards against a stalled UI (or pathological tiny
                        // delays) making us spin through the whole animation.
                        let mut steps = 0;
                        while steps < anim.frames.len()
                            && anim.frame_started.elapsed() >= anim.frames[anim.current].delay
                        {
                            anim.frame_started += anim.frames[anim.current].delay;
                            anim.current = (anim.current + 1) % anim.frames.len();
                            changed = true;
                            steps += 1;
                        }
                        // If we were so far behind we hit the cap, resync to "now" so we
                        // don't keep racing to catch up frame-by-frame.
                        if steps == anim.frames.len() {
                            anim.frame_started = Instant::now();
                        }
                        if changed {
                            let frame_image = anim.frames[anim.current].image.clone();
                            image.texture = Some(ctx.load_texture(
                                "anim_frame",
                                frame_image.clone(),
                                Default::default(),
                            ));
                            // Keep full_res_image pointed at the displayed frame so
                            // clipboard copies grab what the user actually sees.
                            image.full_res_image = frame_image;
                        }
                        let remaining = anim.frames[anim.current]
                            .delay
                            .saturating_sub(anim.frame_started.elapsed());
                        ctx.request_repaint_after(remaining);
                    } else if anim.pending.is_some() {
                        // Only the first frame is decoded so far; poll again shortly to
                        // pick up the rest as the background decoder produces them.
                        ctx.request_repaint_after(Duration::from_millis(8));
                    }
                }

                let available_rect = ui.available_rect_before_wrap();
                let response = ui.allocate_rect(available_rect, egui::Sense::click_and_drag());

                let full_res_size = if let Some(svg) = image.svg.as_ref() {
                    Vec2::new(svg.intrinsic[0], svg.intrinsic[1])
                } else {
                    Vec2::new(image.full_res_image.width() as f32, image.full_res_image.height() as f32)
                };

                // Handle Scale to Fit / Default fit.
                //
                // Two auto modes, both recompute every frame (so the view adapts as the
                // full-res decode swaps in or the window resizes) until the user takes
                // manual control:
                //  - is_scaled_to_fit ("Scale to fit" checkbox ON): zoom to fill the
                //    viewport, even zooming IN for images smaller than the viewport.
                //  - !interacted (default, checkbox OFF): "actual size or zoomed-out-
                //    to-fit" — zoom capped at 1.0 so no image is ever zoomed in on
                //    open; large images shrink to fit, small images show at native
                //    size (letterboxed). Once the user pans or zooms, `interacted`
                //    becomes true and the zoom/offset are pinned (manual mode).
                let aspect_ratio = full_res_size.x / full_res_size.y;
                let available_aspect = available_rect.width() / available_rect.height();
                let mut fit_size = available_rect.size();
                if aspect_ratio > available_aspect {
                    fit_size.y = fit_size.x / aspect_ratio;
                } else {
                    fit_size.x = fit_size.y * aspect_ratio;
                }
                let fit_zoom = fit_size.x / full_res_size.x;
                if self.is_scaled_to_fit {
                    self.zoom = fit_zoom;
                    self.offset = (available_rect.size() - fit_size) / 2.0;
                    // Kill velocity when in fit mode
                    self.velocity = Vec2::ZERO;
                } else if !self.interacted {
                    // Default: actual size, or zoomed out to fit if larger. Never
                    // zoomed in (zoom capped at 1.0).
                    self.zoom = fit_zoom.min(1.0);
                    let displayed = full_res_size * self.zoom;
                    self.offset = (available_rect.size() - displayed) / 2.0;
                    self.velocity = Vec2::ZERO;
                }

                let mut is_interacting = false;
                // True only when offset actually moved due to a *pan* (drag or fling
                // velocity) this frame — never set for scroll-zoom. The viewport
                // clamp uses this to constrain manual throwing without fighting
                // cursor-anchored zoom (which must keep the hovered pixel fixed).
                let mut panned = false;

                // Handle Dragging & Inertia
                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    // Pre-dampen the drag delta per axis so the image can't be pushed
                    // past a viewport wall mid-drag — the drag eases into the wall
                    // instead of accumulating an out-of-bounds offset that a later
                    // clamp would snap back. The free-axis component is untouched, so
                    // the pan stays responsive everywhere except at the walls.
                    let dampened = dampen_drag_delta(
                        delta,
                        self.offset,
                        full_res_size * self.zoom,
                        available_rect.size(),
                    );
                    self.offset += dampened;
                    // Capture momentum as a smoothed average of recent motion rather than
                    // a single frame's delta. This makes the fling reflect the overall
                    // gesture and stops a brief pause right before release from killing it.
                    self.velocity = self.velocity * 0.4 + dampened * 0.6;
                    self.is_scaled_to_fit = false;
                    self.interacted = true;
                    is_interacting = true;
                    panned = true;
                } else {
                    // Apply velocity to position first (let it slide)
                    self.offset += self.velocity;
                    if self.velocity != Vec2::ZERO {
                        panned = true;
                    }
                }

                // Handle Zooming
                if let Some(hover_pos) = response.hover_pos() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        let old_zoom = self.zoom;
                        let zoom_delta = (scroll / 200.0) * self.zoom;
                        // Don't allow zooming out past the point where the image would be
                        // smaller than 5px on its shortest side. If the image itself is
                        // natively <= 5px, we can't zoom out at all (floor at native zoom).
                        let min_dim = full_res_size.x.min(full_res_size.y);
                        let min_zoom = if min_dim < 5.0 { 1.0 } else { 5.0 / min_dim };
                        self.zoom = (self.zoom + zoom_delta).max(min_zoom);
                        // Anchor the zoom to the image pixel under the cursor. When the
                        // cursor is over a non-image area (letterbox / empty canvas),
                        // clamp to the nearest image pixel so the image stays on screen
                        // and zooms toward the closest image content instead of sliding
                        // off-screen and snapping back via the spring.
                        let mut image_coords =
                            (hover_pos - available_rect.min - self.offset) / old_zoom;
                        image_coords.x = image_coords.x.clamp(0.0, full_res_size.x);
                        image_coords.y = image_coords.y.clamp(0.0, full_res_size.y);
                        self.offset -= image_coords * (self.zoom - old_zoom);
                        self.is_scaled_to_fit = false;
                        self.interacted = true;
                        self.velocity = Vec2::ZERO;
                        is_interacting = true;
                    }
                }

                // Hard viewport constraint (no spring): the image can never be pushed
                // entirely out of view via *manual panning*. This only constrains
                // drag/fling motion (`panned`); it deliberately does NOT run on
                // scroll-zoom frames or on resting idle frames, so cursor-anchored
                // zoom keeps the hovered pixel exactly fixed and the image stays
                // where the zoom leaves it rather than snapping back into bounds.
                if !self.is_scaled_to_fit && panned {
                    let screen_size = available_rect.size();
                    let scaled_image_size = full_res_size * self.zoom;
                    let clamp_axis =
                        |offset: &mut f32, velocity: &mut f32, view_dim: f32, img_dim: f32| {
                            let (lo, hi) = if img_dim >= view_dim {
                                (view_dim - img_dim, 0.0)
                            } else {
                                (0.0, view_dim - img_dim)
                            };
                            if *offset < lo {
                                *offset = lo;
                                *velocity = 0.0;
                            } else if *offset > hi {
                                *offset = hi;
                                *velocity = 0.0;
                            }
                        };
                    clamp_axis(&mut self.offset.x, &mut self.velocity.x, screen_size.x, scaled_image_size.x);
                    clamp_axis(&mut self.offset.y, &mut self.velocity.y, screen_size.y, scaled_image_size.y);
                }

                // Bouncing & Constraints
                if !self.is_scaled_to_fit && !is_interacting {
                    let screen_size = available_rect.size();
                    let scaled_image_size = full_res_size * self.zoom;

                    // Friction-only: decay pan/fling velocity to a stop. The spring
                    // snap-back that used to re-center / re-align the image after zoom
                    // or pan is intentionally removed — the image now stays exactly
                    // where the cursor-anchored zoom or the drag leaves it.
                    let friction = 0.92; // Slipperiness (0.0 - 1.0)
                    let handle_axis =
                        |_offset: &mut f32, velocity: &mut f32, _view_dim: f32, _img_dim: f32| {
                            *velocity *= friction;
                        };

                    handle_axis(&mut self.offset.x, &mut self.velocity.x, screen_size.x, scaled_image_size.x);
                    handle_axis(&mut self.offset.y, &mut self.velocity.y, screen_size.y, scaled_image_size.y);

                    // Stop simulation if movement is negligible to save CPU
                    if self.velocity.length_sq() > 0.01 {
                        ctx.request_repaint();
                    } else {
                        self.velocity = Vec2::ZERO;
                    }
                }

                // We keep repainting as long as the image is moving significantly
                if self.velocity.length_sq() > 0.1 {
                    ctx.request_repaint();
                } else {
                    self.velocity = Vec2::ZERO;
                }

                // (SVG) Vector images are re-rasterized at the current displayed
                // pixel size whenever the zoom drifts far enough that the cached
                // texture would no longer be ~1:1 with the screen. Pan never
                // triggers a re-render (it just shifts the same texture); only a
                // meaningful change to the on-screen pixel size does. The render
                // is capped at min(max_texture_side, SVG_RENDER_CAP) so memory and
                // per-render time stay bounded even under extreme zoom-in; past the
                // cap the GPU upscales the cap-sized raster (still sharp).
                // (SVG) Render ONLY the visible crop, at viewport device
                // resolution, transformed so the on-screen region maps 1:1 into a
                // viewport-sized texture. The texture is therefore never upscaled
                // on draw (the only thing that blurs) at ANY zoom level — even
                // absurd zoom-in, where the visible region is a tiny crop of a
                // conceptually huge vector. Memory is bounded by the screen, never
                // the zoom. Trade-off: pan changes the visible region, so it
                // re-renders; we cache the last render params and skip when nothing
                // moved, so static frames cost nothing.
                if image.svg.is_some() {
                    let ppp = ctx.pixels_per_point();
                    let cap = (self.max_texture_side.min(crate::svg::SVG_RENDER_CAP as usize)) as f32;
                    // Viewport size in device pixels (what the draw rect will cover).
                    let vw = (available_rect.width() * ppp).round().clamp(1.0, cap) as u32;
                    let vh = (available_rect.height() * ppp).round().clamp(1.0, cap) as u32;
                    // Transform: device px = svg_user * s + (tx, ty), where
                    //   s  = zoom * ppp        (device px per SVG user unit)
                    //   tx = offset.x * ppp     (SVG origin's screen position)
                    // so the picture is positioned/scaled exactly as the gesture
                    // math intends for `full_res_size * zoom`.
                    let s = self.zoom * ppp;
                    let tx = self.offset.x * ppp;
                    let ty = self.offset.y * ppp;
                    let need_render = match image.svg.as_ref().and_then(|g| g.render_state.as_ref()) {
                        None => true,
                        Some(r) => {
                            r.size != [vw, vh]
                                || (r.s - s).abs() > 0.5
                                || (r.tx - tx).abs() > 0.5
                                || (r.ty - ty).abs() > 0.5
                        }
                    };
                    if need_render {
                        let rendered = {
                            let svg = image.svg.as_ref().expect("svg guarded above");
                            crate::svg::render_svg_viewport(&svg.data, vw, vh, s, tx, ty)
                        };
                        match rendered {
                            Ok(ci) => {
                                image.texture = Some(ctx.load_texture(
                                    "svg_raster",
                                    ci.clone(),
                                    Default::default(),
                                ));
                                // Keep full_res_image synced so clipboard copies
                                // grab the currently-displayed raster.
                                image.full_res_image = ci;
                                if let Some(g) = image.svg.as_mut() {
                                    g.render_state =
                                        Some(SvgRenderState { size: [vw, vh], s, tx, ty });
                                }
                            }
                            Err(e) => log::warn!("SVG render failed: {}", e),
                        }
                    }
                }

                // (c) Tiled images (larger than the GPU's max texture side) always
                // tile directly from the full-res bitmap. Non-tiled images upload
                // the whole full-res image as a single texture and draw it at any
                // zoom (GPU sampling).
                let show_tiles = image.needs_tiling;

                if !show_tiles {
                    if !image.tile_cache.is_empty() {
                        log::debug!("Zoomed out, clearing tile cache of {} textures.", image.tile_cache.len());
                        image.tile_cache.clear();
                    }

                    let svg_mode = image.svg.is_some();
                    // SVG: the viewport-crop texture already covers exactly the
                    // viewport (the transform landed the visible region 1:1).
                    // Other images: draw the texture at the gesture-computed rect.
                    let image_rect = if svg_mode {
                        available_rect
                    } else {
                        let scaled_size = full_res_size * self.zoom;
                        Rect::from_min_size(available_rect.min + self.offset, scaled_size)
                    };
                    if let Some(tex) = image.texture.as_ref() {
                        ui.painter().image(
                            tex.id(),
                            image_rect,
                            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            Color32::WHITE,
                        );
                        let pv = tex.size_vec2();
                        prof_paint_milestone(pv.x as usize, pv.y as usize);
                    }
                } else {
                    let screen_offset_in_image_pixels = (available_rect.min - (available_rect.min + self.offset)) / self.zoom;
                    let screen_size_in_image_pixels = available_rect.size() / self.zoom;
                    let visible_image_rect = Rect::from_min_size(
                        Pos2::new(screen_offset_in_image_pixels.x, screen_offset_in_image_pixels.y),
                        screen_size_in_image_pixels,
                    );

                    let min_col_f = visible_image_rect.min.x / TILE_SIZE as f32;
                    let max_col_f = visible_image_rect.max.x / TILE_SIZE as f32;
                    let min_row_f = visible_image_rect.min.y / TILE_SIZE as f32;
                    let max_row_f = visible_image_rect.max.y / TILE_SIZE as f32;

                    // Clamp the tile loop bounds to the actual tile grid of the image to prevent visual glitches.
                    let num_cols = (image.full_res_image.width() + TILE_SIZE - 1) / TILE_SIZE;
                    let num_rows = (image.full_res_image.height() + TILE_SIZE - 1) / TILE_SIZE;

                    let row_start = (min_row_f.floor() as i32).max(0) as usize;
                    let row_end = (max_row_f.ceil() as i32).max(0) as usize;
                    let col_start = (min_col_f.floor() as i32).max(0) as usize;
                    let col_end = (max_col_f.ceil() as i32).max(0) as usize;

                    for row in row_start..row_end.min(num_rows) {
                        for col in col_start..col_end.min(num_cols) {
                            let tile_key = (row, col);

                            // Get both texture and dimensions from cache, or create and cache both.
                            let (texture_id, tile_dims) = if let Some((texture, dims)) = image.tile_cache.get(&tile_key) {
                                (texture.id(), *dims)
                            } else {
                                let x_start = col * TILE_SIZE;
                                let y_start = row * TILE_SIZE;
                                // Calculate the actual width and height of this tile, clamping to image edges
                                let tile_w = (x_start + TILE_SIZE).min(image.full_res_image.width()) - x_start;
                                let tile_h = (y_start + TILE_SIZE).min(image.full_res_image.height()) - y_start;

                                if tile_w == 0 || tile_h == 0 { continue; }

                                // Manually copy the pixel data row by row
                                let mut tile_pixels = Vec::with_capacity(tile_w * tile_h);
                                for y in 0..tile_h {
                                    let src_y = y_start + y;
                                    let row_start_index = src_y * image.full_res_image.width();
                                    let row_slice_start = row_start_index + x_start;
                                    tile_pixels.extend_from_slice(&image.full_res_image.pixels[row_slice_start..row_slice_start + tile_w]);
                                }

                                let tile_image = ColorImage { size: [tile_w, tile_h], pixels: tile_pixels, source_size: Vec2::new(tile_w as f32, tile_h as f32) };

                                let texture = ctx.load_texture(format!("tile_{}_{}", row, col), tile_image, Default::default());
                                let id = texture.id();
                                let dims = [tile_w, tile_h];
                                image.tile_cache.insert(tile_key, (texture, dims));
                                (id, dims)
                            };

                            let tile_min_in_image_pixels = Pos2::new((col * TILE_SIZE) as f32, (row * TILE_SIZE) as f32);
                            let tile_min_on_screen = available_rect.min + self.offset + tile_min_in_image_pixels.to_vec2() * self.zoom;

                            // Use the actual tile dimensions for drawing, not the fixed TILE_SIZE.
                            let tile_dims_vec = Vec2::new(tile_dims[0] as f32, tile_dims[1] as f32);
                            let tile_screen_rect = Rect::from_min_size(tile_min_on_screen, tile_dims_vec * self.zoom);

                            if available_rect.intersects(tile_screen_rect) {
                                ui.painter().image(texture_id, tile_screen_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                            }
                        }
                    }
                    prof_paint_milestone(full_res_size.x as usize, full_res_size.y as usize);
                }

                let scaled_size = full_res_size * self.zoom;
                let image_screen_rect = Rect::from_min_size(available_rect.min + self.offset, scaled_size);
                if ui.clip_rect().intersects(image_screen_rect) {
                    ui.painter().add(Shape::Rect(RectShape::stroke(image_screen_rect, 0.0, (1.0, Color32::from_gray(80)), egui::StrokeKind::Outside)));
                }

                response.context_menu(|ui| {
                    if ui.checkbox(&mut self.is_fullscreen, "Fullscreen (F)").clicked() {
                        ui.close();
                    };
                    if ui.checkbox(&mut self.is_scaled_to_fit, "Scale to fit (Enter)").clicked() {
                        ui.close();
                    };
                    if ui.checkbox(&mut self.is_randomized, "Random order").clicked() {
                        if self.is_randomized {
                            let current_image_index = self.image_order[self.current_index];
                            #[allow(deprecated)]
                            let mut rng = rand::rng();
                            use rand::seq::SliceRandom;
                            self.image_order.shuffle(&mut rng);
                            if let Some(pos) = self.image_order.iter().position(|&i| i == current_image_index) {
                                self.current_index = pos;
                            }
                        } else {
                            let current_image_index = self.image_order[self.current_index];
                            self.image_order = (0..self.image_files.len()).collect();
                            if let Some(pos) = self.image_order.iter().position(|&i| i == current_image_index) {
                                self.current_index = pos;
                            }
                        }
                        ui.close();
                    };
                });

            } else if let Some(err) = &self.last_error {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new(err).color(Color32::RED).size(18.0));
                });
            } else if self.full_res_pending {
                let current_path = self
                    .image_files
                    .get(self.image_order.get(self.current_index).copied().unwrap_or(usize::MAX));
                let label = match current_path {
                    Some(p) => format!("Loading {}…", p.display()),
                    None => "Loading…".to_string(),
                };
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new(label).color(Color32::from_gray(180)).size(18.0));
                });
            }
        });

    if self.show_delete_confirmation {
            let path = self.image_files.get(self.image_order[self.current_index]).cloned();
        egui::Window::new("Delete File")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(&ctx, |ui| {
                if let Some(path) = &path {
                ui.label(format!("Are you sure you want to delete '{}'?", path.display()));
                ui.add_space(10.0);
                let confirm_with_enter = ctx.input(|i| i.key_pressed(egui::Key::Enter));
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.show_delete_confirmation = false;
                    }
                    if ui.button(egui::RichText::new("Delete").color(Color32::RED)).clicked() || confirm_with_enter {
                        if let Err(e) = fs::remove_file(path) {
                            self.last_error = Some(format!("Failed to delete file: {}", e));
                        } else {
                            log::info!("Deleted file: {}", path.display());
                            let removed_order_index = self.image_order.remove(self.current_index);
                            self.image_files.remove(removed_order_index);
                            for order_idx in self.image_order.iter_mut() {
                                if *order_idx > removed_order_index {
                                    *order_idx -= 1;
                                }
                            }
                            if self.image_files.is_empty() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            } else {
                                self.current_index %= self.image_files.len();
                                self.load_image_at_index(self.current_index, &ctx);
                            }
                        }
                        self.show_delete_confirmation = false;
                    }
                });
                }
            });
        }
    }
}

impl Drop for ImageViewerApp {
    fn drop(&mut self) {
        self.shutdown_workers();
    }
}
