{
  lib,
  pythonSet,
  workspace,
  libGL,
  libGLU,
  xorg,
  version ? "unstable",
}:

(pythonSet.mkVirtualEnv "sustainable-foraging-env" workspace.deps.default).overrideAttrs (_: {
  pname = "sustainable-foraging";
  inherit version;


  meta = {
    description = "Sustainable Foraging with PettingZoo";
    homepage = "https://github.com/pixel-87/sustainable-foraging";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ pixel-87 ];
    mainProgram = "sustainable-foraging";
  };
})
