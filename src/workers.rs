use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{channel, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use crate::decode::load_full_for_worker;
use crate::types::{FullResReply, FullResRequest, FullResWorker, MemoryGate};

/// Spawn the full-res worker dispatcher. The dispatcher is a lightweight thread
/// that accepts requests and (memory permitting) immediately spawns a fresh
/// decode thread for each, so the user's *latest* navigation starts decoding
/// right away instead of waiting for a previously-started decode to finish. A
/// shared generation counter lets stale decodes (whose result the user no
/// longer cares about) drop their results when they eventually finish. The
/// `MemoryGate` caps concurrency so a burst of rapid navigation can't OOM the
/// system on large RAW/FITS files.
pub fn spawn_full_res_worker(ctx: egui::Context, gate: Arc<MemoryGate>) -> FullResWorker {
    let (req_tx, req_rx) = channel::<FullResRequest>();
    let (reply_tx, reply_rx) = channel::<FullResReply>();
    thread::spawn(move || full_res_dispatcher_loop(req_rx, reply_tx, ctx, gate));
    FullResWorker { tx: req_tx, rx: reply_rx }
}

fn full_res_dispatcher_loop(
    req_rx: Receiver<FullResRequest>,
    reply_tx: Sender<FullResReply>,
    ctx: egui::Context,
    gate: Arc<MemoryGate>,
) {
    // Bumped on every accepted request; a decode thread only sends its reply if
    // its generation still matches when the decode finishes.
    let generation = Arc::new(AtomicU64::new(0));
    loop {
        let mut req = match req_rx.recv() {
            Ok(r) => r,
            Err(_) => return, // App dropped — exit.
        };
        // While we wait for a memory slot, keep draining newer requests so the
        // decode we eventually start is for the freshest navigation target.
        loop {
            while let Ok(newer) = req_rx.try_recv() {
                req = newer;
            }
            if gate.try_acquire() {
                break;
            }
            log::debug!("Foreground decoder waiting for memory slot");
            thread::sleep(Duration::from_millis(100));
        }
        let my_gen = generation.fetch_add(1, Ordering::Relaxed) + 1;
        let reply_tx = reply_tx.clone();
        let ctx2 = ctx.clone();
        let generation2 = generation.clone();
        let gate2 = gate.clone();
        thread::spawn(move || {
            let worker_t0 = Instant::now();
            eprintln!("[prof] T+{:.1}ms worker: decode thread started for '{}'", crate::app::proc_ms(), req.path.display());

            // Full-resolution decode, nothing else: no reduced-resolution
            // preview is ever produced. The UI shows a "Loading…" placeholder
            // until this reply arrives, then swaps the sharp image in directly.
            let t_full = Instant::now();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load_full_for_worker(&req.path)))
                .unwrap_or_else(|_| Err(format!("decoder panicked for {}", req.path.display())));
            eprintln!(
                "[prof] T+{:.1}ms worker: full-res decode done in {:.1}ms (worker thread total {:.1}ms)",
                crate::app::proc_ms(),
                t_full.elapsed().as_secs_f64() * 1000.0,
                worker_t0.elapsed().as_secs_f64() * 1000.0
            );
            gate2.release();
            // If a newer request arrived while we were decoding, this result is
            // stale; drop it — the latest navigation's decode will replace it.
            if generation2.load(Ordering::Relaxed) != my_gen {
                log::debug!("Discarding stale decode result for {}", req.path.display());
                return;
            }
            let result = match outcome {
                Ok(loaded) => Ok(loaded),
                Err(e) => Err(e),
            };
            let reply = FullResReply {
                path: req.path.clone(),
                result,
            };
            if reply_tx.send(reply).is_ok() {
                ctx2.request_repaint();
            }
        });
    }
}
