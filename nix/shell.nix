{
  mkShell,
  callPackage,
  stdenv,

  # python tooling
  pythonSet,
  workspace,

  uv,
  ty,
  ruff,

  # graphics 
  xvfb-run,
  zlib,
  zstd,
  libGL,
  libGLU,
  xorg,
  freetype,
  fontconfig,
}:

let
  defaultPackage = callPackage ./default.nix { };
  virtualenv = pythonSet.mkVirtualEnv "sustainable-foraging-dev-env" workspace.deps.all;
in
mkShell {
  inputsFrom = [ virtualenv ];

  packages = [
    uv
    ty 
    ruff
    xvfb-run
  ];

  env = {
    UV_NO_SYNC = "1";
    UV_PYTHON = pythonSet.python.interpreter;
    UV_PYTHON_DOWNLOADS = "never";
  };

  shellHook = ''
    unset PYTHONPATH
    export REPO_ROOT=$(git rev-parse --show-toplevel)
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:${zlib}/lib:${zstd.out}/lib:${libGL}/lib:${libGLU}/lib:${xorg.libX11}/lib:${xorg.libXcursor}/lib:${xorg.libXi}/lib:${xorg.libXinerama}/lib:${freetype}/lib:${fontconfig.lib}/lib:$LD_LIBRARY_PATH
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
  '';

}
