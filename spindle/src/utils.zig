const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn free3dMatrix(allocator: Allocator, matrix: [][][]f32) void {
    for (matrix) |*first_dims| {
        for (first_dims.*) |*second_dims| {
            allocator.free(second_dims.*);
        }
        allocator.free(first_dims.*);
    }
    allocator.free(matrix);
}

pub fn free2dMatrix(allocator: Allocator, matrix: [][]f32) void {
    for (matrix) |*first_dims| {
        allocator.free(first_dims.*);
    }
    allocator.free(matrix);
}

pub fn elapsedTime(allocator: Allocator, start_time: i64) ![]const u8 {
    const elapsed_time: u64 = @intCast(std.time.timestamp() - start_time);
    const spm: u64 = 60; // Number of seconds per minute
    return try std.fmt.allocPrint(allocator, "{}:{d:0>2}", .{
        @divFloor(elapsed_time, spm),
        @rem(elapsed_time, spm),
    });
}
