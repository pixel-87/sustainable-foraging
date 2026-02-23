{
  description = "Sustainable Foraging with PettingZoo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs  = { self, nixpkgs, ... }:
  let
    forAllSystems = 
      f: nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed (
        system: f nixpkgs.legacyPackages.${system}
      );
  in 
  {
    packages = forAllSystems (pkgs: {
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.sustainable-foraging;
      sustainable-foraging = pkgs.callPackage ./nix/default.nix { version = self.shortRev or "unstable"; };
      });

    overlays.default = final: _: {
      sustainable-foraging = final.callPackage ./nix/default.nix { version = self.shortRev or "unstable"; };
    };

    devShells = forAllSystems (pkgs: {
      default = pkgs.callPackage ./nix/shell.nix { };
      });
  };

}
