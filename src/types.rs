// --- Advanced Data Structures for Tiled Viewing ---
use egui::{ColorImage, TextureHandle};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, Sender},
    },
    time::{Duration, Instant},
};

pub struct DisplayableImage {
    /// The full-resolution original image, kept in CPU memory.
    pub full_res_image: ColorImage,
    /// Single GPU texture holding the currently-displayed image. For a still
    /// raster this is the whole full-res upload; for SVGs it holds the most
    /// recently re-rendered viewport raster (regenerated in place as zoom
    /// drifts); for animated images it holds the active frame. `None` for tiled
    /// images, which render directly from full-res tiles.
    pub texture: Option<TextureHandle>,
    /// Cache for detail tiles to avoid re-uploading them to the GPU every frame.
    pub tile_cache: HashMap<(usize, usize), (TextureHandle, [usize; 2])>,
    /// Does this image actually need tiling, or is it small enough to fit on the GPU?
    pub needs_tiling: bool,
    /// Animation playback state for animated images (e.g. GIFs). `None` for stills.
    pub animation: Option<Animation>,
    /// Present only for vector (SVG) images: the raw SVG bytes plus intrinsic
    /// size. When set, the draw loop treats `full_res_image`-derived sizing as
    /// the SVG's intrinsic extent (so all zoom/pan/fit math behaves like a
    /// raster of that size) and regenerates `texture` at the current
    /// displayed resolution for crisp-at-any-zoom rendering.
    pub svg: Option<SvgSource>,
}

/// Vector source for an SVG image: the uncompressed bytes plus the intrinsic
/// pixel extent parsed from its `width`/`height`/`viewBox`. Kept on the
/// `DisplayableImage` so the UI thread can re-rasterize at any zoom.
pub struct SvgSource {
    pub data: Vec<u8>,
    /// `[width, height]` in SVG user units — drives fit aspect and (at zoom=1)
    /// means one user unit equals one screen pixel.
    pub intrinsic: [f32; 2],
    /// Cached parameters of the last viewport-crop render. When the current
    /// viewport size + transform (zoom/offset) match these within a sub-pixel
    /// epsilon, the existing texture is reused; otherwise the visible region is
    /// re-rasterized. `None` forces a render on the first frame.
    pub render_state: Option<SvgRenderState>,
}

/// Snapshot of the inputs that produced the current `texture` for an
/// SVG, used to decide whether a re-render is needed this frame.
pub struct SvgRenderState {
    pub size: [u32; 2],
    pub s: f32,
    pub tx: f32,
    pub ty: f32,
}

/// Request sent to the long-lived full-res decoder worker.
pub struct FullResRequest {
    pub path: PathBuf,
}

/// Reply produced by the full-res worker for the UI to consume.
pub struct FullResReply {
    pub path: PathBuf,
    pub result: Result<LoadedImage, String>,
}

/// Single long-lived worker that decodes the foreground full-res image.
/// Rapid navigation queues many requests; the worker drains intermediate ones
/// and only decodes the most recently requested image, so the CPU isn't split
/// across N stale decodes when the user settles on a frame.
pub struct FullResWorker {
    pub tx: Sender<FullResRequest>,
    pub rx: Receiver<FullResReply>,
}

/// Memory-aware admission control for image decoders. Holds a counter of how
/// many decodes are currently running and a sysinfo handle for checking RAM.
///
/// Rapid navigation can otherwise spawn many concurrent decoders (each peaking
/// hundreds of MB for large RAW/FITS files) and OOM the box — so before either
/// the foreground dispatcher or a bulk-preload worker begins a decode they must
/// `try_acquire()` a slot from the gate, which adapts to available memory.
pub struct MemoryGate {
    active: AtomicUsize,
    system: Mutex<sysinfo::System>,
}

impl MemoryGate {
    pub fn new() -> Self {
        Self {
            active: AtomicUsize::new(0),
            system: Mutex::new(sysinfo::System::new()),
        }
    }

    /// Reserve a decode slot if memory permits. Returns false when the system
    /// is too tight to safely start another decode; callers should sleep and
    /// retry.
    pub fn try_acquire(&self) -> bool {
        loop {
            let active = self.active.load(Ordering::Acquire);
            let max = self.max_concurrent();
            if active >= max {
                return false;
            }
            if self
                .active
                .compare_exchange(active, active + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }

    pub fn release(&self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }

    /// How many concurrent decodes we'll tolerate right now. Always >= 1 so the
    /// app never deadlocks even when memory is critically low; on a low-memory
    /// system this collapses to "one decode at a time", restoring the previous
    /// non-preempting behavior.
    fn max_concurrent(&self) -> usize {
        let Ok(mut sys) = self.system.lock() else { return 1 };
        sys.refresh_memory();
        let available = sys.available_memory();
        let total = sys.total_memory();
        drop(sys);

        // Reserve a safety margin so we don't push the OS into swap.
        let safety = (total / 10).max(512 * 1024 * 1024);
        let usable = available.saturating_sub(safety);
        // RAW/FITS decoding peaks at ~300 MB for typical files; use that as a
        // budget per concurrent decode.
        const PER_DECODE_BYTES: u64 = 300 * 1024 * 1024;
        let by_memory = (usable / PER_DECODE_BYTES) as usize;
        // Cap at a reasonable ceiling so we don't spin up dozens even on a
        // workstation with hundreds of GB.
        by_memory.clamp(1, 8)
    }
}

/// One frame of an animated image (e.g. GIF), already composited to the full
/// canvas, plus how long it should be shown before advancing.
pub struct AnimationFrame {
    pub image: ColorImage,
    pub delay: Duration,
}

/// Playback state for an animated image. Frame 0 is decoded up front so it can
/// be shown immediately; the remaining frames arrive lazily over `pending` from
/// a background decoder (which is `None` once all frames are decoded, or for
/// sources that decode eagerly). The UI drains `pending` each tick, advances
/// `current` based on wall-clock time, and re-uploads the active frame to the
/// displayable image's `texture` whenever it changes.
pub struct Animation {
    pub frames: Vec<AnimationFrame>,
    pub current: usize,
    /// When the currently-displayed frame began showing.
    pub frame_started: Instant,
    /// Receives the not-yet-decoded frames from a background thread, in order.
    /// `None` once the stream ends (all frames decoded) or for eager sources.
    pub pending: Option<Receiver<AnimationFrame>>,
}

// Simplified enum for loaded image data before GPU upload
pub enum LoadedImage {
    Static(ColorImage),
    /// An animation. `frames` holds the frames decoded so far (always at least
    /// the first); `pending` streams the remaining frames in from a background
    /// decoder (`None` once complete, or for eagerly-decoded sources like GIF).
    Animated {
        frames: Vec<AnimationFrame>,
        pending: Option<Receiver<AnimationFrame>>,
    },
    /// SVG: raw bytes + intrinsic size. Rasterization is deferred to the UI
    /// thread (which knows the current zoom) so it can render at display size for
    /// crisp-at-any-zoom output. The worker therefore just hands the bytes over.
    Svg { data: Vec<u8>, intrinsic: [f32; 2] },
}
