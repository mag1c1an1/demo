{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
        ];

        shellHook = ''
          echo "Python dev shell"
          echo "Python: $(python --version)"
          echo "uv: $(uv --version 2>/dev/null || true)"
        '';
      };
    };
}
