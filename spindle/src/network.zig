const std = @import("std");
const math = @import("math.zig");
const Dataset = @import("dataset.zig").Dataset;

// See: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
pub const Network = struct {
    const Self = @This();
    weights: [][][]f32, // index of current layer, index of neuron in current layer, index of neuron in the next layer
    biases: [][]f32, // index of current layer minus 1 (layer 0 a.k.a input doesn't have weights), index of neuron in current layer
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, prng: *std.rand.DefaultPrng, layer_sizes: []const u32) !Network {
        const layers_num = layer_sizes.len;
        const weights: [][][]f32 = try allocator.alloc([][]f32, layers_num - 1);
        const biases: [][]f32 = try allocator.alloc([]f32, layers_num - 1);
        for (weights, biases, 0..) |*layer_weights, *layer_biases, layer_index| {
            layer_weights.* = try allocator.alloc([]f32, layer_sizes[layer_index]);
            layer_biases.* = try allocator.alloc(f32, layer_sizes[layer_index + 1]);
            for (layer_weights.*) |*next_layer| {
                next_layer.* = try allocator.alloc(f32, layer_sizes[layer_index + 1]);
                for (next_layer.*) |*weight| {
                    weight.* = prng.random().floatNorm(f32);
                }
            }
            for (layer_biases.*) |*bias| {
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
        self.free(self.weights, self.biases);
    }

    pub fn train(self: Self, dataset: Dataset, batch_size: u32, epochs: u32, learning_rate: f32) !void {
        for (0..epochs) |epoch| {
            const batches = try dataset.getBatches(batch_size);
            defer {
                for (batches.x_batches, batches.y_batches) |x_batches, y_batches| {
                    for (x_batches, y_batches) |x, y| {
                        self.allocator.free(x);
                        self.allocator.free(y);
                    }
                    self.allocator.free(x_batches);
                    self.allocator.free(y_batches);
                }
                self.allocator.free(batches.x_batches);
                self.allocator.free(batches.y_batches);
            }
            for (batches.x_batches, batches.y_batches) |x_batches, y_batches| {
                const weight_gradients: [][][]f32 = try self.allocator.alloc([][]f32, self.weights.len);
                const bias_gradients: [][]f32 = try self.allocator.alloc([]f32, self.biases.len);
                for (
                    weight_gradients,
                    bias_gradients,
                    self.weights,
                    self.biases,
                ) |*layer_weight_gradients, *layer_bias_gradients, layer_weights, layer_biases| {
                    layer_weight_gradients.* = try self.allocator.alloc([]f32, layer_weights.len);
                    layer_bias_gradients.* = try self.allocator.alloc(f32, layer_biases.len);
                    for (layer_weight_gradients.*, layer_weights) |*next_layer_neuron_gradients, next_layer_neurons| {
                        next_layer_neuron_gradients.* = try self.allocator.alloc(f32, next_layer_neurons.len);
                        for (next_layer_neuron_gradients.*) |*gradient| {
                            gradient.* = 0.0;
                        }
                    }
                    for (layer_bias_gradients.*) |*gradient| {
                        gradient.* = 0.0;
                    }
                }
                defer self.free(weight_gradients, bias_gradients);
                for (x_batches, y_batches) |x, y| {
                    const gradients = try self.backprop(x, y);
                    defer self.free(gradients.weight_gradients, gradients.bias_gradients);
                    for (
                        weight_gradients,
                        bias_gradients,
                        gradients.weight_gradients,
                        gradients.bias_gradients,
                    ) |*layer_total_weight_gradients, *layer_total_bias_gradients, layer_weight_gradients, layer_bias_gradients| {
                        for (
                            layer_total_weight_gradients.*,
                            layer_weight_gradients,
                        ) |*next_layer_total_neuron_gradients, next_layer_neuron_gradients| {
                            for (
                                next_layer_total_neuron_gradients.*,
                                next_layer_neuron_gradients,
                            ) |*total_weight_gradient, weight_gradient| {
                                total_weight_gradient.* += weight_gradient;
                            }
                        }
                        for (
                            layer_total_bias_gradients.*,
                            layer_bias_gradients,
                        ) |*total_bias_gradient, bias_gradient| {
                            total_bias_gradient.* += bias_gradient;
                        }
                    }
                }
                for (
                    self.weights,
                    self.biases,
                    weight_gradients,
                    bias_gradients,
                ) |*layer_weights, *layer_biass, layer_total_weight_gradients, layer_total_bias_gradients| {
                    for (layer_weights.*, layer_total_weight_gradients) |*next_layer_neurons, next_layer_total_neuron_gradients| {
                        for (next_layer_neurons.*, next_layer_total_neuron_gradients) |*weight, total_weight_gradient| {
                            weight.* -= learning_rate * total_weight_gradient / @as(f32, @floatFromInt(batch_size));
                        }
                    }
                    for (layer_biass.*, layer_total_bias_gradients) |*bias, total_bias_gradient| {
                        bias.* -= learning_rate * total_bias_gradient / @as(f32, @floatFromInt(batch_size));
                    }
                }
            }
            const loss = try self.evaluate(dataset);
            std.debug.print("Epoch {}: loss={d:.10}\n", .{ epoch, loss });
        }
    }

    pub fn evaluate(self: Self, dataset: Dataset) !f32 {
        var loss: f32 = 0.0;
        for (dataset.xs, dataset.ys) |x, y| {
            const layers = try self.feedforward(x);
            const a = layers.as[layers.as.len - 1];
            loss += cost(a, y);
        }
        loss /= @floatFromInt(dataset.xs.len);
        return loss;
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
            for (zs, as) |*z, *a| {
                self.allocator.free(z.*);
                self.allocator.free(a.*);
            }
            self.allocator.free(zs);
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

    fn cost(a: []f32, y: []f32) f32 {
        if (a.len != y.len) {
            unreachable;
        }
        var c: f32 = 0.0;
        for (a, y) |ai, yi| {
            c += std.math.pow(f32, ai - yi, 2);
        }
        c /= 2.0;
        return c;
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

    fn free(self: Self, weights: [][][]f32, biases: [][]f32) void {
        for (weights, biases) |*layer_weights, *layer_biases| {
            for (layer_weights.*) |*next_layer| {
                self.allocator.free(next_layer.*);
            }
            self.allocator.free(layer_weights.*);
            self.allocator.free(layer_biases.*);
        }
        self.allocator.free(weights);
        self.allocator.free(biases);
    }
};
