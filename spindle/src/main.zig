const std = @import("std");
const Dataset = @import("dataset.zig").Dataset;
const Network = @import("network.zig").Network;

const INPUT_SIZE = 2;
const OUTPUT_SIZE = 1;
const SAMPLES_LEN = 4;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
    // Dataset
    var raw_xs = [SAMPLES_LEN][INPUT_SIZE]f32{
        [_]f32{ 0, 0 },
        [_]f32{ 0, 1 },
        [_]f32{ 1, 0 },
        [_]f32{ 1, 1 },
    };
    var raw_ys = [SAMPLES_LEN][OUTPUT_SIZE]f32{
        [_]f32{0},
        [_]f32{1},
        [_]f32{1},
        [_]f32{0},
    };
    const xs: [][]f32 = try allocator.alloc([]f32, SAMPLES_LEN);
    const ys: [][]f32 = try allocator.alloc([]f32, SAMPLES_LEN);
    for (xs, ys, 0..) |*x, *y, i| {
        x.* = &raw_xs[i];
        y.* = &raw_ys[i];
    }
    defer {
        for (xs, ys) |*x, *y| {
            allocator.free(x.*);
            allocator.free(y.*);
        }
        allocator.free(xs);
        allocator.free(ys);
    }
    const train_dataset = Dataset.init(allocator, &prng, xs, ys);
    // Network
    const network = try Network.init(allocator, &prng, &[_]u32{ INPUT_SIZE, 3, OUTPUT_SIZE });
    defer network.deinit();
    try network.train(train_dataset, 2, 1000, 1);
}
