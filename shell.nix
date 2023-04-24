{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [ (pkgs.python310.withPackages (p: with p; [
  black
  gym
  gymnasium
  matplotlib
  numpy
  ])) ];

  shellHook = ''
    export PYTHONPATH=$PYTHONPATH:$(readlink -f .)
  '';
}