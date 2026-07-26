pkgname=nimbleview-git
pkgver=59bdd73
pkgrel=1
pkgdesc="Fast cross-platform image viewer written in Rust, supporting many raw camera formats"
arch=('x86_64')
url="https://github.com/markg85/NimbleView"
license=('GPL-2.0-or-later')
depends=(
  'dav1d'
  'libgl'
  'libx11'
  'libxcb'
  'libxcursor'
  'libxi'
  'libxrender'
  'libxkbcommon'
  'wayland'
  'hicolor-icon-theme'
)
makedepends=('rust' 'cargo' 'git' 'clang' 'cmake' 'meson' 'nasm')
provides=('nimbleview')
conflicts=('nimbleview')
# strip: ship a stripped binary (the system makepkg.conf sets !strip).
# !lto: the native C/C++ dependencies (thorvg, libjpeg-turbo) are compiled with
# makepkg's CFLAGS; under the lto option those become GCC LTO bitcode that
# rust-lld cannot read, so disable LTO for this package.
options=(strip !lto)
source=("git+$url.git#branch=main")
sha256sums=('SKIP')

pkgver() {
  cd NimbleView
  git rev-parse --short HEAD
}

prepare() {
  cd NimbleView
  export RUSTUP_TOOLCHAIN=stable
  cargo fetch --locked --target "$(rustc --print host-tuple)"
}

build() {
  cd NimbleView
  export RUSTUP_TOOLCHAIN=stable
  export CARGO_TARGET_DIR=target
  # No debug info in the packaged binary, and remap the build directory so the
  # binary carries no reference to $srcdir (cleaner, reproducible output).
  export CARGO_PROFILE_RELEASE_DEBUG=0
  export RUSTFLAGS="--remap-path-prefix=$srcdir=."
  cargo build --release --frozen
}

package() {
  cd NimbleView
  install -Dm755 target/release/nimbleview "$pkgdir/usr/bin/nimbleview"
  install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
  install -Dm644 nimbleview.desktop "$pkgdir/usr/share/applications/nimbleview.desktop"
  install -Dm644 NimbleView.svg "$pkgdir/usr/share/icons/hicolor/scalable/apps/nimbleview.svg"
}
