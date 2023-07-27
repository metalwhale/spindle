const std = @import("std");
const math = @import("math.zig");
const Dataset = @import("dataset.zig").Dataset;

// See: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
pub const Network = struct {
    const Self = @This();
    // 0: input layer, L: output layer, l: current layer, k: previous layer, m: next layer
    // Since the layer 0 doesn't have weights, indices of weights and biases start counting from layer 1
    weights: [][][]f32,
    biases: [][]f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, prng: *std.rand.DefaultPrng, layer_sizes: []const u32) !Network {
        const layers_num = layer_sizes.len;
        const weights: [][][]f32 = try allocator.alloc([][]f32, layers_num - 1);
        const biases: [][]f32 = try allocator.alloc([]f32, layers_num - 1);
        for (weights, biases, 0..) |*wk, *bl, l| {
            wk.* = try allocator.alloc([]f32, layer_sizes[l]);
            bl.* = try allocator.alloc(f32, layer_sizes[l + 1]);
            for (wk.*) |*wl| {
                wl.* = try allocator.alloc(f32, layer_sizes[l + 1]);
                for (wl.*) |*w| {
                    w.* = prng.random().floatNorm(f32);
                }
            }
            for (bl.*) |*b| {
                b.* = prng.random().floatNorm(f32);
            }
        }
        const network = Network{ .weights = weights, .biases = biases, .allocator = allocator };
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
                // Calculate batched gradients
                const d_ws: [][][]f32 = try self.allocator.alloc([][]f32, self.weights.len);
                const d_bs: [][]f32 = try self.allocator.alloc([]f32, self.biases.len);
                for (d_ws, d_bs, self.weights, self.biases) |*d_wk, *d_bl, wk, bl| {
                    d_wk.* = try self.allocator.alloc([]f32, wk.len);
                    d_bl.* = try self.allocator.alloc(f32, bl.len);
                    for (d_wk.*, wk) |*d_wl, wl| {
                        d_wl.* = try self.allocator.alloc(f32, wl.len);
                        for (d_wl.*) |*d_w| {
                            d_w.* = 0.0;
                        }
                    }
                    for (d_bl.*) |*d_b| {
                        d_b.* = 0.0;
                    }
                }
                defer self.free(d_ws, d_bs);
                for (x_batches, y_batches) |x, y| {
                    const gradients = try self.backprop(x, y);
                    defer self.free(gradients.d_ws, gradients.d_bs);
                    for (d_ws, d_bs, gradients.d_ws, gradients.d_bs) |*d_wk, *d_bl, per_d_wk, per_d_bl| {
                        for (d_wk.*, per_d_wk) |*d_wl, per_d_wl| {
                            for (d_wl.*, per_d_wl) |*d_w, per_d_w| {
                                d_w.* += per_d_w / @as(f32, @floatFromInt(batch_size));
                            }
                        }
                        for (d_bl.*, per_d_bl) |*d_b, per_d_b| {
                            d_b.* += per_d_b / @as(f32, @floatFromInt(batch_size));
                        }
                    }
                }
                // Update weights and biases
                for (self.weights, self.biases, d_ws, d_bs) |*wk, *bl, d_wk, d_bl| {
                    for (wk.*, d_wk) |*wl, d_wl| {
                        for (wl.*, d_wl) |*w, d_w| {
                            w.* -= learning_rate * d_w;
                        }
                    }
                    for (bl.*, d_bl) |*b, d_b| {
                        b.* -= learning_rate * d_b;
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
        for (self.weights, self.biases, zs, as, 0..) |wl, bl, *z, *a, l| {
            const ak: []f32 = if (l == 0) x else as[l - 1];
            // TODO: Check if every sub-slice in the 2nd dimension of b have the same size
            const weighted_sum = try self.allocator.alloc(f32, wl[0].len);
            defer self.allocator.free(weighted_sum);
            for (weighted_sum, 0..) |*s, j| {
                const weight = try self.allocator.alloc(f32, ak.len);
                defer self.allocator.free(weight);
                for (weight, 0..) |*w, i| {
                    w.* = wl[i][j];
                }
                s.* = try math.dot(self.allocator, ak, weight);
            }
            z.* = try math.add(self.allocator, weighted_sum, bl);
            a.* = try math.sigmoid(self.allocator, z.*);
        }
        return .{ .zs = zs, .as = as };
    }

    // See: http://neuralnetworksanddeeplearning.com/images/tikz21.png
    //   δL = ∇aC ⊙ σ'(zL)
    //   δl = ((wm)T * δm) ⊙ σ'(zl)
    //   ∂C/∂bl = δl
    //   ∂C/∂wl = ak * δl
    fn backprop(self: Self, x: []f32, y: []f32) !struct { d_ws: [][][]f32, d_bs: [][]f32 } {
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
        const d_ws: [][][]f32 = try self.allocator.alloc([][]f32, self.weights.len); // ∂C/∂w
        const d_bs: [][]f32 = try self.allocator.alloc([]f32, self.biases.len); // ∂C/∂b
        for (1..d_bs.len + 1) |reverse_index| {
            const l = d_bs.len - reverse_index; // Reverse iteration starts from the last layer
            var d_l: []f32 = undefined; // δl
            if (l == d_bs.len - 1) { // Last layer
                const d_a = try self.costDerivative(as[l], y); // ∇aC
                defer self.allocator.free(d_a);
                const d_z = try math.sigmoidDerivative(self.allocator, zs[l]); // σ'(zL)
                defer self.allocator.free(d_z);
                d_l = try math.mul(self.allocator, d_a, d_z); // δL = ∇aC * σ'(zL)
            } else {
                const wm = self.weights[l + 1];
                const d_lm = d_bs[l + 1];
                const weighted_sum = try self.allocator.alloc(f32, wm.len);
                defer self.allocator.free(weighted_sum);
                for (wm, weighted_sum) |w, *s| {
                    // NOTE: We don't need to use transpose here since we choose (k, l) as the order of dimensions in the weights
                    // in contrast to (l, k) on the website
                    s.* = try math.dot(self.allocator, w, d_lm); // (wm)T * δm
                }
                const d_z = try math.sigmoidDerivative(self.allocator, zs[l]); // σ'(zl)
                defer self.allocator.free(d_z);
                d_l = try math.mul(self.allocator, weighted_sum, d_z); // δl = ((wm)T * δm) ⊙ σ'(zl)
            }
            d_bs[l] = d_l; // ∂C/∂bl = δl
            const ak = if (l == 0) x else as[l - 1];
            const d_w = try self.allocator.alloc([]f32, ak.len); // ∂C/∂wl
            for (ak, d_w) |aki, *d_wi| { // ∂C/∂wl = ak * δl
                d_wi.* = try self.allocator.alloc(f32, d_l.len);
                for (d_l, d_wi.*) |d_lj, *d_wij| {
                    d_wij.* = aki * d_lj;
                }
            }
            d_ws[l] = d_w;
        }
        return .{ .d_ws = d_ws, .d_bs = d_bs };
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
        for (weights, biases) |*wk, *bl| {
            for (wk.*) |*wl| {
                self.allocator.free(wl.*);
            }
            self.allocator.free(wk.*);
            self.allocator.free(bl.*);
        }
        self.allocator.free(weights);
        self.allocator.free(biases);
    }
};
