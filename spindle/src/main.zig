const std = @import("std");
const Network = @import("network.zig").Network;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const network = try Network.init(allocator, &[_]u16{ 2, 3, 1 });
    _ = network;
}
