use derive_more::{Display, Error, From};
use std::io::{self, Read, Seek, SeekFrom, Write};

#[derive(Debug)]
pub struct Img3Header {
    pub magic: u32,
    pub full_size: u32,
    pub ident: u32,
}

#[derive(Debug)]
pub struct Img3TagInfo {
    pub magic: u32,
    pub total_length: u32,
    pub data_length: u32,
    pub data_offset: u64,
}

/// Stream-oriented representation of an IMG3 file.
/// R must implement Read + Seek (files, cursor over bytes, etc).
pub struct Img3Stream<R: Read + Seek> {
    reader: R,
    pub header: Img3Header,
    // offset where tags start (typically 20)
    tags_start: u64,
    // next tag offset to read (advances as you call `next_tag`)
    next_tag_offset: u64,
    // file/stream length as determined on construction (if available)
    file_len: u64,
}

#[derive(Debug, Display, Error, From)]
pub enum Img3Error {
    #[display("Data section not found")]
    DataSectionNotFound,
    #[display("File too small")]
    FileTooSmall,
    #[display("Invalid magic")]
    InvalidMagic,
    #[display("Invalid tag magic")]
    InvalidTagMagic,
    #[display("Invalid tag length")]
    InvalidTagLength,
    #[display("I/O error: {source}")]
    Io { source: std::io::Error },
}

impl<R: Read + Seek> Img3Stream<R> {
    /// Construct an `Img3Stream` from a seekable reader. Reads only the top-level header.
    pub fn new(mut reader: R) -> Result<Self, Img3Error> {
        // Read header (20 bytes)
        let mut header_buf = [0u8; 20];
        reader.read_exact(&mut header_buf)?;

        let magic =
            u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]);
        let full_size =
            u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]);
        // data_size and skip_dist are present in header but we don't need them here
        let _data_size =
            u32::from_le_bytes([header_buf[8], header_buf[9], header_buf[10], header_buf[11]]);
        let _skip_dist = u32::from_le_bytes([
            header_buf[12],
            header_buf[13],
            header_buf[14],
            header_buf[15],
        ]);
        let ident = u32::from_le_bytes([
            header_buf[16],
            header_buf[17],
            header_buf[18],
            header_buf[19],
        ]);

        if header_buf.len() < 20 {
            return Err(Img3Error::FileTooSmall);
        }

        let header = Img3Header {
            magic,
            full_size,
            ident,
        };

        let current_pos = reader.stream_position()?;
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(current_pos))?;

        let tags_start = 20u64;
        let next_tag_offset = tags_start;

        Ok(Img3Stream {
            reader,
            header,
            tags_start,
            next_tag_offset,
            file_len,
        })
    }

    /// Read the next tag header and return `Img3TagInfo`.
    /// This reads only the 12-byte per-tag header and computes data offset; it does not
    /// load the tag payload into memory.
    pub fn next_tag(&mut self) -> Result<Option<Img3TagInfo>, Img3Error> {
        if self.next_tag_offset + 12 > self.file_len {
            return Ok(None);
        }

        self.reader.seek(SeekFrom::Start(self.next_tag_offset))?;

        let mut tag_hdr = [0u8; 12];
        self.reader.read_exact(&mut tag_hdr)?;

        let tag_magic = u32::from_le_bytes([tag_hdr[0], tag_hdr[1], tag_hdr[2], tag_hdr[3]]);
        let total_length = u32::from_le_bytes([tag_hdr[4], tag_hdr[5], tag_hdr[6], tag_hdr[7]]);
        let data_length = u32::from_le_bytes([tag_hdr[8], tag_hdr[9], tag_hdr[10], tag_hdr[11]]);

        if total_length < 12 {
            return Err(Img3Error::InvalidTagLength);
        }
        let next_offset = self.next_tag_offset + u64::from(total_length);
        if next_offset > self.file_len {
            return Err(Img3Error::InvalidTagLength);
        }

        let data_offset = self.next_tag_offset + 12;

        self.next_tag_offset = next_offset;

        Ok(Some(Img3TagInfo {
            magic: tag_magic,
            total_length,
            data_length,
            data_offset,
        }))
    }

    /// Stream the raw bytes of a tag (payload only) into the given writer.
    /// This seeks to the tag's data offset and uses `io::copy` with `Take`, so the payload
    /// is streamed without a full in-memory allocation.
    pub fn copy_tag_data<W: Write>(
        &mut self,
        tag: &Img3TagInfo,
        writer: &mut W,
    ) -> Result<u64, Img3Error> {
        self.reader.seek(SeekFrom::Start(tag.data_offset))?;
        let mut take = self.reader.by_ref().take(tag.data_length as u64);
        let copied = io::copy(&mut take, writer)?;
        Ok(copied)
    }

    /// Find the first "DATA" tag (magic 0x44415441) and stream it into the writer.
    pub fn stream_data_section<W: Write>(&mut self, writer: &mut W) -> Result<u64, Img3Error> {
        self.next_tag_offset = self.tags_start;

        while let Some(tag) = self.next_tag()? {
            if tag.magic == 0x44415441 {
                return Ok(self.copy_tag_data(&tag, writer)?);
            }
        }
        Err(Img3Error::DataSectionNotFound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // small synthetic IMG3: header (20 bytes) + one tag with "DATA" and payload "hello"
    fn make_small_img3() -> Vec<u8> {
        let mut v = Vec::new();
        // magic, full_size, data_size, skip_dist, ident (all little endian)
        v.extend_from_slice(&0x12345678u32.to_le_bytes()); // magic
        v.extend_from_slice(&0u32.to_le_bytes()); // full_size (ignored for test)
        v.extend_from_slice(&0u32.to_le_bytes()); // data_size
        v.extend_from_slice(&0u32.to_le_bytes()); // skip_dist
        v.extend_from_slice(&0x9u32.to_le_bytes()); // ident

        // tag header: magic "DATA", total_len = 12 + 5, data_len = 5
        v.extend_from_slice(&0x44415441u32.to_le_bytes()); // "DATA"
        v.extend_from_slice(&(12u32 + 5u32).to_le_bytes()); // total length
        v.extend_from_slice(&5u32.to_le_bytes()); // data length
        // tag data: "hello"
        v.extend_from_slice(b"hello");
        v
    }

    #[test]
    fn test_stream_basic() {
        let data = make_small_img3();
        let cursor = Cursor::new(data);
        let mut s = Img3Stream::new(cursor).expect("construct");
        let mut out = Vec::new();
        // iterate tags
        while let Some(tag) = s.next_tag().expect("next_tag") {
            let mut buf = Vec::new();
            s.copy_tag_data(&tag, &mut buf).expect("copy");
            // only one tag with "hello"
            assert_eq!(buf, b"hello");
            out.extend_from_slice(&buf);
        }
        assert_eq!(out, b"hello");
    }
}
