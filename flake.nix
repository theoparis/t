{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      nixpkgs,
      ...
    }:
    let
      eachSystem = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
        "riscv64-linux"
      ];
    in
    {
      devShells = eachSystem (
        system:
        with (import nixpkgs {
          inherit system;
          overlays = [
            (final: previous: {
              python3 = previous.python3.override {
                packageOverrides = pyfinal: pyprev: {
                  torch = previous.python3Packages.torchWithRocm;
                };
              };
            })
          ];
        }); {
          default = mkShell rec {
            nativeBuildInputs = [
              pyrefly
              rustup
              pkg-config
              nixfmt
              nil
              ruff
              (python3.withPackages (
                pythonPkgs: with pythonPkgs; [
                  transformers
                  sentencepiece
                  protobuf
                  trl
                ]
              ))
            ];

            buildInputs = [
              rocmPackages.clr
            ];

            LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
          };
        }
      );
    };
}
