use derive_more::Error;
use derive_more::derive::Display;
use rootcause::Report;
use rootcause::prelude::ResultExt;
use std::fs::File;
use std::io::Read;

use crate::img3::{Img3Error, Img3Stream};

pub mod decryptor;
pub mod img3;

#[derive(Debug, Display, Error)]
pub enum DeviceTreeParseError {
    #[display("Unexpected end of data")]
    UnexpectedEndOfData,
    #[display("Invalid property name")]
    InvalidPropertyName,
    #[display("Invalid property value")]
    InvalidPropertyValue,
    #[display("Invalid node name")]
    InvalidNodeName,
}

#[derive(Debug, Display, Error)]
pub enum DeviceTreeError {
    #[display("I/O error: {_0}")]
    Io(std::io::Error),
    #[display("Device tree parse error: {_0}")]
    Parse(DeviceTreeParseError),
    #[display("Invalid IMG3 file")]
    InvalidImg3File(Img3Error),
}

impl From<std::io::Error> for DeviceTreeError {
    fn from(err: std::io::Error) -> Self {
        DeviceTreeError::Io(err)
    }
}

#[derive(Debug)]
struct DeviceTreeNode {
    name: String,
    properties: Vec<DeviceTreeProperty>,
    children: Vec<DeviceTreeNode>,
}

#[derive(Debug)]
struct DeviceTreeProperty {
    name: String,
    value: Vec<u8>,
}

fn parse_node(data: &[u8], offset: &mut usize) -> Result<DeviceTreeNode, DeviceTreeError> {
    if *offset + 8 > data.len() {
        return Err(DeviceTreeError::Parse(
            DeviceTreeParseError::UnexpectedEndOfData,
        ));
    }

    let num_props = u32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    let num_children = u32::from_le_bytes([
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;

    let mut properties = Vec::new();
    for _ in 0..num_props {
        if *offset + 32 + 4 > data.len() {
            break;
        }

        let name = String::from_utf8_lossy(&data[*offset..*offset + 32])
            .trim_end_matches('\0')
            .to_string();
        *offset += 32;

        let value_len = u32::from_le_bytes([
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
        ]);
        *offset += 4;

        if *offset + value_len as usize > data.len() {
            break;
        }

        let value = data[*offset..*offset + value_len as usize].to_vec();
        *offset += value_len as usize;
        *offset = (*offset + 3) & !3;

        properties.push(DeviceTreeProperty { name, value });
    }

    let mut children = Vec::new();
    for _ in 0..num_children {
        if let Ok(child) = parse_node(data, offset) {
            children.push(child);
        }
    }

    let name = properties
        .iter()
        .find(|p| p.name == "name")
        .map(|p| {
            String::from_utf8_lossy(&p.value)
                .trim_end_matches('\0')
                .to_string()
        })
        .unwrap_or_else(|| "unnamed".to_string());

    Ok(DeviceTreeNode {
        name,
        properties,
        children,
    })
}
