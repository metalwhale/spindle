const std = @import("std");
const math = @import("math.zig");

// See: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
pub const Network = struct {
    const Self = @This();
    weights: [][][]f32, // index of current layer, index of neuron in current layer, index of neuron in the next layer
    biases: [][]f32, // index of current layer minus 1 (layer 0 a.k.a input doesn't have weights), index of neuron in current layer
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, prng: *std.rand.DefaultPrng, layer_sizes: []const u16) !Network {
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

    fn feedforward(self: Self, x: []f32) !struct { zs: [][]f32, as: [][]f32 } {
        const zs: [][]f32 = try self.allocator.alloc([]f32, self.biases.len);
        const as: [][]f32 = try self.allocator.alloc([]f32, self.biases.len);
        for (self.weights, self.biases, zs, as, 0..) |layer_weights, layer_biases, *z, *a, layer_index| {
            const ak: []f32 = if (layer_index == 0) x else as[layer_index - 1];
            // TODO: Check if every sub-slice in the 2nd dimension of b have the same size
            const weighted_sum = try self.allocator.alloc(f32, layer_weights[0].len);
            defer self.allocator.free(weighted_sum);
            for (weighted_sum, 0..) |*s, j| {
                const weight = try self.allocator.alloc(f32, ak.len);
                defer self.allocator.free(weight);
                for (weight, 0..) |*w, i| {
                    w.* = layer_weights[i][j];
                }
                s.* = try math.dot(self.allocator, ak, weight);
            }
            z.* = try math.add(self.allocator, weighted_sum, layer_biases);
            a.* = try math.sigmoid(self.allocator, z.*);
        }
        return .{ .zs = zs, .as = as };
    }

    // See: http://neuralnetworksanddeeplearning.com/images/tikz21.png
    //   δL = ∇aC ⊙ σ'(zL)
    //   δl = ((wm)T * δm) ⊙ σ'(zl)
    //   ∂C/∂bl = δl
    //   ∂C/∂wl = ak * δl
    // L: last layer, l: current layer, k: previous layer, m: next layer
    fn backprop(self: Self, x: []f32, y: []f32) !struct { weight_gradients: [][][]f32, bias_gradients: [][]f32 } {
        // Feedforward
        const layers = try self.feedforward(x);
        const zs = layers.zs;
        const as = layers.as;
        defer {
            for (zs) |*z| {
                self.allocator.free(z.*);
            }
            self.allocator.free(zs);
            for (as) |*a| {
                self.allocator.free(a.*);
            }
            self.allocator.free(as);
        }
        // Backprop
        const weight_gradients: [][][]f32 = try self.allocator.alloc([][]f32, self.weights.len); // ∂C/∂w
        const bias_gradients: [][]f32 = try self.allocator.alloc([]f32, self.biases.len); // ∂C/∂b
        for (1..bias_gradients.len + 1) |reverse_index| {
            const layer_index = bias_gradients.len - reverse_index; // Reverse iteration starts from the last layer
            var d_l: []f32 = undefined; // δl
            if (layer_index == bias_gradients.len - 1) { // Last layer
                const d_a = try self.costDerivative(as[layer_index], y); // ∇aC
                defer self.allocator.free(d_a);
                const d_z = try math.sigmoidDerivative(self.allocator, zs[layer_index]); // σ'(zL)
                defer self.allocator.free(d_z);
                d_l = try math.mul(self.allocator, d_a, d_z); // δL = ∇aC * σ'(zL)
            } else {
                const wm = self.weights[layer_index + 1];
                const d_lm = bias_gradients[layer_index + 1];
                const weighted_sum = try self.allocator.alloc(f32, wm.len);
                defer self.allocator.free(weighted_sum);
                for (wm, weighted_sum) |w, *s| {
                    // NOTE: We don't need to use transpose here since we choose (k, l) as the order of dimensions in the weights
                    // in contrast to (l, k) on the website
                    s.* = try math.dot(self.allocator, w, d_lm); // (wm)T * δm
                }
                const d_z = try math.sigmoidDerivative(self.allocator, zs[layer_index]); // σ'(zl)
                defer self.allocator.free(d_z);
                d_l = try math.mul(self.allocator, weighted_sum, d_z); // δl = ((wm)T * δm) ⊙ σ'(zl)
            }
            bias_gradients[layer_index] = d_l; // ∂C/∂bl = δl
            const ak = if (layer_index == 0) x else as[layer_index - 1];
            const d_w = try self.allocator.alloc([]f32, ak.len); // ∂C/∂wl
            for (ak, d_w) |aki, *d_wi| { // ∂C/∂wl = ak * δl
                d_wi.* = try self.allocator.alloc(f32, d_l.len);
                for (d_l, d_wi.*) |d_lj, *d_wij| {
                    d_wij.* = aki * d_lj;
                }
            }
            weight_gradients[layer_index] = d_w;
        }
        return .{ .weight_gradients = weight_gradients, .bias_gradients = bias_gradients };
    }

    // Cost function: C = 1/2 * Σ(yi - ai)^2 (mean squared error)
    // => Derivative of cost function dC/dai = ai - yi
    fn costDerivative(self: Self, a: []f32, y: []f32) ![]f32 {
        if (a.len != y.len) {
            unreachable;
        }
        const dc = try self.allocator.alloc(f32, a.len);
        for (dc, a, y) |*dci, ai, yi| {
            dci.* = ai - yi;
        }
        return dc;
    }
};
