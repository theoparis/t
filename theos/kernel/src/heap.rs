use linked_list_allocator::LockedHeap;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

pub fn init_heap(heap_start: u64, heap_size: usize) {
    crate::kprintln!(
        "GRAVITY HEAP: Initializing at {:x} (size {:x})",
        heap_start,
        heap_size
    );
    unsafe {
        ALLOCATOR.lock().init(heap_start as *mut u8, heap_size);
    }
}
