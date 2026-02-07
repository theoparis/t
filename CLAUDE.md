# CLAUDE.md

This file provides guidance for interacting with the monorepo codebase.

## Build and Test

Run the commands in a `nix develop` shell to get access to the necessary tools.

## Updating Rust Dependencies

```bash
reindeer buckify
```

### Building the Emulator

```bash
buck2 build '//zed-os/tools/emulator:ios-emulator'
```

## Workspace Structure

- `kernel/` - Main kernel source code
- `libs/` - Shared libraries
- `tools/` - Utility tools
- `rootfs/` - Root filesystem
