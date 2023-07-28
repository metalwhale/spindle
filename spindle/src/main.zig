const std = @import("std");
const Mnist = @import("mnist.zig").Mnist;
const Network = @import("network.zig").Network;
const Config = @import("network.zig").Config;

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
    // Dataset
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 2) {
        std.debug.print("Usage: {s} <dir_path>\n", .{args[0]});
        std.os.exit(1);
    }
    const dir_path = args[1];
    var mnist = try Mnist.init(allocator);
    defer mnist.deinit();
    const datasets = try mnist.readData(dir_path);
    const train_dataset = datasets.train_dataset;
    const val_dataset = datasets.val_dataset;
    const sizes = train_dataset.getSizes();
    // Network
    const layer_sizes = &[_]usize{ sizes.input_size, 30, sizes.output_size };
    const network = try Network.init(allocator, &prng, layer_sizes);
    defer network.deinit();
    const config = Config{ .batch_size = 10, .epochs = 15, .learning_rate = 3 };
    try network.train(train_dataset, val_dataset, config);
}
