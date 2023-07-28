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
