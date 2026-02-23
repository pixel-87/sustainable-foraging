{
  lib,
  python3Packages,
  libGL,
  libGLU,
  xorg,
  version ? "unstable",
}:

python3Packages.buildPythonApplication {
  pname = "sustainable-foraging";
  inherit version;

  src = lib.cleanSource ../.;

  pyproject = true;

  buildInputs = [
  libGL
  libGLU
  xorg.libX11
  xorg.libXcursor
  xorg.libXinerama
  xorg.libXi
];

  dependencies = with python3Packages; [ ];

  nativeCheckInputs = [ python3Packages.pytestCheckHook ];

  meta = {
    description = "Sustainable Foraging with PettingZoo";
    homepage = "https://github.com/pixel-87/sustainable-foraging";
    license = lib.licenses.gpl3Plus;
    maintainers = with lib.maintainers; [ pixel-87 ];
    mainProgram = "sustainable-foraging";
  };
}
