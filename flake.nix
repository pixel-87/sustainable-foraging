{
  description = "Sustainable Foraging with PettingZoo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs  = { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems }:
  let
    inherit (nixpkgs) lib legacyPackages;

    forAllSystems = 
      f: lib.genAttrs lib.systems.flakeExposed (
        system: f {
          inherit system;
          pkgs = legacyPackages.${system};
        }
      );

    workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

    overlay = workspace.mkPyprojectOverlay {
      sourcePreference = "wheel";
    };

    editableOverlay = workspace.mkEditablePyprojectOverlay {
      root = "$REPO_ROOT";
    };

    pythonSets = forAllSystems ({ system, pkgs }:
      let
        python = pkgs.python312;
      in 
      (pkgs.callPackage pyproject-nix.build.packages { inherit python;}).overrideScope
        (lib.composeManyExtensions [ 
          pyproject-build-systems.overlays.wheel
          overlay
        ])
    );
  in 
  {
    packages = forAllSystems ({ system, pkgs }: {
      sustainable-foraging = pkgs.callPackage ./nix/default.nix {
        pythonSet = pythonSets.${system};
        inherit workspace;
        version = self.shortRev or "unstable"; 
      };
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.sustainable-foraging;
    });

    overlays.default = final: _: {
      sustainable-foraging = final.callPackage ./nix/default.nix {
        pythonSet = pythonSets.${final.stdenv.hostPlatform.system};
        inherit workspace;
        version = self.shortRev or "unstable"; 
      };
    };

    devShells = forAllSystems ({ system, pkgs }: {
      default = pkgs.callPackage ./nix/shell.nix {
        pythonSet = pythonSets.${system}.overrideScope editableOverlay;
        inherit workspace;
      };
    });
  };

}
