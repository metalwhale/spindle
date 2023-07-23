const std = @import("std");
const math = @import("math.zig");
const Allocator = std.mem.Allocator;

// See: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
pub const Network = struct {
    const Self = @This();
    weights: [][][]f32, // index of current layer, index of neuron in current layer, index of neuron in the next layer
    biases: [][]f32, // index of current layer minus 1 (layer 0 a.k.a input doesn't have weights), index of neuron in current layer
    allocator: Allocator,

    pub fn init(allocator: std.mem.Allocator, layer_sizes: []const u16) !Network {
        var prng = std.rand.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
        const layers_num = layer_sizes.len;
        // Initialize weights
        const weights: [][][]f32 = try allocator.alloc([][]f32, layers_num - 1);
        for (weights, 0..) |*layer, layer_index| {
            layer.* = try allocator.alloc([]f32, layer_sizes[layer_index]);
            for (layer.*) |*next_layer| {
                next_layer.* = try allocator.alloc(f32, layer_sizes[layer_index + 1]);
                for (next_layer.*) |*weight| {
                    weight.* = prng.random().floatNorm(f32);
                }
            }
        }
        // Initialize biases
        const biases: [][]f32 = try allocator.alloc([]f32, layers_num - 1);
        for (biases, 0..) |*layer, layer_index| {
            layer.* = try allocator.alloc(f32, layer_sizes[layer_index + 1]);
            for (layer.*) |*bias| {
                bias.* = prng.random().floatNorm(f32);
            }
        }
        const network = Network{
            .weights = weights,
            .biases = biases,
            .allocator = allocator,
        };
        return network;
    }

    pub fn deinit(self: Self) void {
        for (self.weights) |*layer| {
            for (layer.*) |*next_layer| {
                self.allocator.free(next_layer.*);
            }
            self.allocator.free(layer.*);
        }
        self.allocator.free(self.weights);
        for (self.biases) |*layer| {
            self.allocator.free(layer.*);
        }
        self.allocator.free(self.biases);
    }

    fn feedforward(self: Self, x: []f32) !void {
        var a = try self.allocator.alloc(f32, x.len);
        std.mem.copy(f32, a, x); // TODO: Directly use x then free it properly instead of copying
        for (self.weights, self.biases) |layer_weights, layer_biases| {
            // TODO: Check if every sub-slice in the 2nd dimension of b have the same size
            const weighted_sum = try self.allocator.alloc(f32, layer_weights[0].len);
            defer self.allocator.free(weighted_sum);
            for (weighted_sum, 0..) |*s, j| {
                const weight = try self.allocator.alloc(f32, a.len);
                defer self.allocator.free(weight);
                for (weight, 0..) |*w, i| {
                    w.* = layer_weights[i][j];
                }
                s.* = try math.dot(self.allocator, a, weight);
            }
            const z = try math.add(self.allocator, weighted_sum, layer_biases);
            defer self.allocator.free(z);
            self.allocator.free(a);
            a = try math.sigmoid(self.allocator, z);
        }
    }
};
