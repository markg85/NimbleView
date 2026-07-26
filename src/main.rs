use eframe::egui;
use std::{env, error::Error, path::PathBuf, time::Instant};

use crate::app::proc_ms;

mod app;
mod decode;
mod formats;
mod svg;
mod types;
mod workers;

use crate::app::ImageViewerApp;
use crate::decode::get_absolute_path;

// --- Main Entry Point ---
fn main() -> Result<(), Box<dyn Error>> {
    // T0: captured before *anything* else so all startup milestones are measured
    // against the true process start.
    crate::app::PROC_START.set(Instant::now()).ok();
    eprintln!("[prof] T+{:.1}ms process start", proc_ms());

    env_logger::init();
    eprintln!("[prof] T+{:.1}ms logger initialized", proc_ms());
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} [/windowed] <imagefile>", args[0]);
        return Ok(());
    }

    let mut is_fullscreen = true;
    let mut image_file_arg = &args[1];

    if args[1].eq_ignore_ascii_case("/windowed") {
        if args.len() > 2 {
            is_fullscreen = false;
            image_file_arg = &args[2];
        } else {
            println!("Missing image file after /windowed");
            return Ok(());
        }
    }

    let initial_path: PathBuf = get_absolute_path(image_file_arg)?;
    eprintln!("[prof] T+{:.1}ms args parsed, path resolved", proc_ms());

    // perf-optimize iteration #2: pre-decode the startup image in parallel with
    // eframe's window/GL setup. run_native spends ~50ms creating the window/GL
    // context before App::new runs; this thread decodes (~20ms) during that gap
    // so the result is ready by the time App::new consumes it. The receiver is
    // handed to ImageViewerApp::new; the worker path polls it directly.
    let (predecode_tx, predecode_rx) =
        std::sync::mpsc::channel::<crate::decode::PredecodeResult>();
    {
        let pd_path = initial_path.clone();
        let pd_tx = predecode_tx;
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let r = crate::decode::load_full_for_worker(&pd_path);
            eprintln!(
                "[prof] T+{:.1}ms predecode: load_full_for_worker done in {:.1}ms (parallel with window setup)",
                crate::app::proc_ms(),
                t0.elapsed().as_secs_f64() * 1000.0
            );
            let _ = pd_tx.send(r);
        });
        // predecode_tx dropped here; only the receiver is kept.
    }
    let predecode = Some((initial_path.clone(), predecode_rx));

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([300.0, 200.0])
	    .with_app_id("nimbleview"),
        ..Default::default()
    };

    eprintln!("[prof] T+{:.1}ms entering eframe::run_native (event loop + window creation begins)", proc_ms());
    eframe::run_native(
        "Nimble View (egui)",
        native_options,
        Box::new(move |cc| Ok(Box::new(ImageViewerApp::new(cc, Some(initial_path), is_fullscreen, predecode)))),
    )?;

    Ok(())
}
