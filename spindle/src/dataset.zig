const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Dataset = struct {
    const Self = @This();
    xs: [][]f32,
    ys: [][]f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, xs: [][]f32, ys: [][]f32) Dataset {
        // TODO: Check if every sample in xs and ys has the same size
        if (xs.len != ys.len) {
            unreachable;
        }
        const dataset = Dataset{ .xs = xs, .ys = ys, .allocator = allocator };
        return dataset;
    }

    pub fn getSizes(self: Self) struct { samples_len: usize, input_size: usize, output_size: usize } {
        return .{ .samples_len = self.xs.len, .input_size = self.xs[0].len, .output_size = self.ys[0].len };
    }

    pub fn getBatches(
        self: Self,
        prng: *std.rand.DefaultPrng,
        batch_size: u32,
    ) !struct { x_batches: [][][]f32, y_batches: [][][]f32 } {
        if (batch_size > self.xs.len) {
            unreachable;
        }
        const indices = try self.allocator.alloc(usize, self.xs.len);
        defer self.allocator.free(indices);
        for (indices, 0..) |*index, i| {
            index.* = i;
        }
        prng.random().shuffle(usize, indices);
        var batches_len = self.xs.len / batch_size;
        if (self.xs.len % batch_size > 0) {
            batches_len += 1;
        }
        const x_batches = try self.allocator.alloc([][]f32, batches_len);
        const y_batches = try self.allocator.alloc([][]f32, batches_len);
        for (x_batches, y_batches, 0..) |*x_batch, *y_batch, batch_index| {
            const current_batch_size = @min(self.xs.len - batch_index * batch_size, batch_size);
            x_batch.* = try self.allocator.alloc([]f32, current_batch_size);
            y_batch.* = try self.allocator.alloc([]f32, current_batch_size);
            for (x_batch.*, y_batch.*, 0..) |*x, *y, i| {
                const data_index = batch_index * batch_size + i;
                const source_x = self.xs[indices[data_index]];
                const source_y = self.ys[indices[data_index]];
                x.* = try self.allocator.alloc(f32, source_x.len);
                y.* = try self.allocator.alloc(f32, source_y.len);
                std.mem.copy(f32, x.*, source_x);
                std.mem.copy(f32, y.*, source_y);
            }
        }
        return .{ .x_batches = x_batches, .y_batches = y_batches };
    }
};
