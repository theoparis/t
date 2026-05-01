use aes::cipher::{Block, BlockDecryptMut, KeyIvInit}; // Import Block and the trait
use std::io::{self, Write};

type Aes128CbcDec = cbc::Decryptor<aes::Aes128>;

pub struct DecryptWriter<W: Write> {
    inner: W,
    decryptor: Aes128CbcDec,
    leftover: Vec<u8>,
}

impl<W: Write> DecryptWriter<W> {
    pub fn new(inner: W, key: &[u8], iv: &[u8]) -> Self {
        Self {
            inner,
            decryptor: Aes128CbcDec::new_from_slices(key, iv).expect("Invalid key/iv"),
            leftover: Vec::with_capacity(16),
        }
    }
}

impl<W: Write> Write for DecryptWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut pos = 0;

        // 1. Handle leftovers (same logic as before)
        if !self.leftover.is_empty() {
            let need = 16 - self.leftover.len();
            let take = std::cmp::min(need, buf.len());
            self.leftover.extend_from_slice(&buf[..take]);
            pos += take;

            if self.leftover.len() == 16 {
                // For a single block, decrypt_block_mut is fine
                let block = Block::<Aes128CbcDec>::from_mut_slice(&mut self.leftover);
                self.decryptor.decrypt_block_mut(block);
                self.inner.write_all(&self.leftover)?;
                self.leftover.clear();
            }
        }

        // 2. Process full blocks
        let remaining = &buf[pos..];
        let num_blocks = remaining.len() / 16;

        if num_blocks > 0 {
            let bytes_to_process = num_blocks * 16;
            let mut chunk = remaining[..bytes_to_process].to_vec();

            // FIX: Convert the &mut [u8] into &mut [Block<Aes128CbcDec>]
            // This is the idiomatic way to handle multiple blocks in the cipher crate
            let blocks = slice_as_chunks_mut::<Aes128CbcDec>(&mut chunk);

            self.decryptor.decrypt_blocks_mut(blocks);

            self.inner.write_all(&chunk)?;
            pos += bytes_to_process;
        }

        if pos < buf.len() {
            self.leftover.extend_from_slice(&buf[pos..]);
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// Helper function to safely cast byte slice to block slice
fn slice_as_chunks_mut<C: BlockDecryptMut>(slice: &mut [u8]) -> &mut [Block<C>] {
    let n = slice.len() / 16;
    let ptr = slice.as_mut_ptr() as *mut Block<C>;
    unsafe { std::slice::from_raw_parts_mut(ptr, n) }
}
