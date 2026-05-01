fn main() -> Result<(), Report> {
    let dt_path = "work/Firmware/all_flash/all_flash.k48ap.production/DeviceTree.k48ap.img3";
    let mut dt_file = File::open(dt_path)?;
    let mut dt_data = Vec::new();
    dt_file.read_to_end(&mut dt_data)?;

    let img3 = Img3File::parse(&dt_data)?;
    let iv = hex::decode("e0a3aa63dae431e573c9827dd3636dd1").unwrap();
    let key =
        hex::decode("50208af7c2de617854635fb4fc4eaa8cddab0e9035ea25abf81b0fa8b0b5654f").unwrap();

    let decrypted_data = img3
        .decrypt_data(&key, iv)
        .ok_or_else(|| DeviceTreeError::Img3DataSectionNotFound)
        .context("Failed to parse IMG3 file")
        .attach(format!("IMG3 file path: {}", dt_path))?;

    // Parse the tree structure
    let mut offset = 0;
    let root = parse_node(&decrypted_data, &mut offset)?;

    // Print the full tree
    print_tree(&root, 0);

    Ok(())
}

fn print_tree(node: &DeviceTreeNode, depth: usize) {
    let indent = "  ".repeat(depth);
    println!("{}{}", indent, node.name);

    if node.name == "chosen" || node.name == "memory-map" {
        for prop in &node.properties {
            println!("{}  prop {}: size={}", indent, prop.name, prop.value.len());
            if prop.value.len() == 8 {
                let v1 = u32::from_le_bytes([
                    prop.value[0],
                    prop.value[1],
                    prop.value[2],
                    prop.value[3],
                ]);
                let v2 = u32::from_le_bytes([
                    prop.value[4],
                    prop.value[5],
                    prop.value[6],
                    prop.value[7],
                ]);
                println!("{}    values: 0x{:08x} 0x{:08x}", indent, v1, v2);
            }
        }
    }

    for prop in &node.properties {
        if prop.name == "reg" && prop.value.len() >= 8 {
            let addr =
                u32::from_le_bytes([prop.value[0], prop.value[1], prop.value[2], prop.value[3]]);
            let size =
                u32::from_le_bytes([prop.value[4], prop.value[5], prop.value[6], prop.value[7]]);
            println!("{}  reg: base=0x{:08x} size=0x{:x}", indent, addr, size);
        }
        if prop.name == "compatible" {
            let s = String::from_utf8_lossy(&prop.value)
                .trim_matches('\0')
                .to_string();
            println!("{}  compatible: {}", indent, s);
        }
    }

    for child in &node.children {
        print_tree(child, depth + 1);
    }
}
