#![allow(dead_code)]

use alloc::vec::Vec;
use lz4_flex::block::{compress, decompress};

/// A simple compressed memory allocator (zram-like).
/// It takes input data (e.g. a page), compresses it using LZ4,
/// and stores it. It returns a handle to the stored data.
pub struct ZAllocator {
    // In a real zsmalloc, we would have size classes and dedicated pages.
    // Here we wrap the system allocator but provide compression.
}

/// Compress data using LZ4.
/// Returns a Vec<u8> containing the compressed data.
pub fn zcompress(data: &[u8]) -> Result<Vec<u8>, &'static str> {
    Ok(compress(data))
}

/// Decompress data using LZ4.
/// LZ4 decompress needs to know the uncompressed size.
/// For zram-like usage, it's usually a fixed block size.
pub fn zdecompress(data: &[u8], uncompressed_size: usize) -> Result<Vec<u8>, &'static str> {
    decompress(data, uncompressed_size).map_err(|_| "Decompression failed")
}

/// A "ZRAM" block device simulator.
/// Stores pages in a compressed format in memory.
pub struct ZRamDevice {
    blocks: Vec<Option<Vec<u8>>>,
    block_size: usize,
}

impl ZRamDevice {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(None);
        }
        Self { blocks, block_size }
    }

    pub fn write_block(&mut self, index: usize, data: &[u8]) -> Result<(), &'static str> {
        if index >= self.blocks.len() || data.len() != self.block_size {
            return Err("Invalid argument");
        }

        // Compress
        let compressed = zcompress(data)?;
        // Store
        self.blocks[index] = Some(compressed);
        Ok(())
    }

    pub fn read_block(&self, index: usize, out: &mut [u8]) -> Result<(), &'static str> {
        if index >= self.blocks.len() || out.len() != self.block_size {
            return Err("Invalid argument");
        }

        if let Some(ref compressed) = self.blocks[index] {
            let decompressed = zdecompress(compressed, self.block_size)?;
            out.copy_from_slice(&decompressed);
            Ok(())
        } else {
            // Block not present, return zeros
            out.fill(0);
            Ok(())
        }
    }
}
