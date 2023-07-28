const std = @import("std");
const math = @import("math.zig");
const utils = @import("utils.zig");
const Dataset = @import("dataset.zig").Dataset;
const Allocator = std.mem.Allocator;

pub const Config = struct {
    batch_size: u32,
    epochs: u32,
    learning_rate: f32,
};

// See: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
// Index name convention:
//   0: input layer, L: output layer, l: current layer, k: previous layer, m: next layer
pub const Network = struct {
    const Self = @This();
    // Since the layer 0 doesn't have weights, indices of weights and biases start pointing from layer 1
    weights: [][][]f32, // Each element represents a 2-d matrix with shape (k, l)
    biases: [][]f32, // Each element represents a 1-d matrix with shape (l)
    allocator: Allocator,
    prng: *std.rand.DefaultPrng,

    pub fn init(allocator: Allocator, prng: *std.rand.DefaultPrng, layer_sizes: []const usize) !Network {
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
        const network = Network{ .weights = weights, .biases = biases, .allocator = allocator, .prng = prng };
        return network;
    }

    pub fn deinit(self: Self) void {
        utils.free3dMatrix(self.allocator, self.weights);
        utils.free2dMatrix(self.allocator, self.biases);
    }

    pub fn train(self: Self, train_dataset: Dataset, val_dataset: Dataset, config: Config) !void {
        var result = try self.evaluate(val_dataset);
        std.debug.print("Before training: loss={d:.10}, acc={d:.2}%\n", .{ result.loss, result.accuracy * 100 });
        const start_time = std.time.timestamp();
        for (0..config.epochs) |epoch| {
            const batches = try train_dataset.getBatches(self.prng, config.batch_size);
            defer {
                utils.free3dMatrix(self.allocator, batches.x_batches);
                utils.free3dMatrix(self.allocator, batches.y_batches);
            }
            for (batches.x_batches, batches.y_batches) |x_batch, y_batch| {
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
                defer {
                    utils.free3dMatrix(self.allocator, d_ws);
                    utils.free2dMatrix(self.allocator, d_bs);
                }
                for (x_batch, y_batch) |x, y| {
                    const gradients = try self.backprop(x, y);
                    defer {
                        utils.free3dMatrix(self.allocator, gradients.d_ws);
                        utils.free2dMatrix(self.allocator, gradients.d_bs);
                    }
                    for (d_ws, d_bs, gradients.d_ws, gradients.d_bs) |*d_wk, *d_bl, per_d_wk, per_d_bl| {
                        for (d_wk.*, per_d_wk) |*d_wl, per_d_wl| {
                            for (d_wl.*, per_d_wl) |*d_w, per_d_w| {
                                d_w.* += per_d_w / @as(f32, @floatFromInt(config.batch_size));
                            }
                        }
                        for (d_bl.*, per_d_bl) |*d_b, per_d_b| {
                            d_b.* += per_d_b / @as(f32, @floatFromInt(config.batch_size));
                        }
                    }
                }
                // Update weights and biases
                for (self.weights, self.biases, d_ws, d_bs) |*wk, *bl, d_wk, d_bl| {
                    for (wk.*, d_wk) |*wl, d_wl| {
                        for (wl.*, d_wl) |*w, d_w| {
                            w.* -= config.learning_rate * d_w;
                        }
                    }
                    for (bl.*, d_bl) |*b, d_b| {
                        b.* -= config.learning_rate * d_b;
                    }
                }
            }
            // Evaluation
            result = try self.evaluate(val_dataset);
            std.debug.print("Epoch {}: elapsed_time={s}, loss={d:.10}, acc={d:.2}%\n", .{
                epoch,
                try utils.elapsedTime(self.allocator, start_time),
                result.loss,
                result.accuracy * 100,
            });
        }
    }

    pub fn evaluate(self: Self, dataset: Dataset) !struct { loss: f32, accuracy: f32 } {
        var loss: f32 = 0.0;
        var correct_count: f32 = 0.0;
        for (dataset.xs, dataset.ys) |x, y| {
            const layers = try self.feedforward(x);
            defer {
                utils.free2dMatrix(self.allocator, layers.zs);
                utils.free2dMatrix(self.allocator, layers.as);
            }
            const aL = layers.as[layers.as.len - 1];
            loss += cost(aL, y);
            if (findMaxIndex(aL) == findMaxIndex(y)) {
                correct_count += 1;
            }
        }
        loss /= @floatFromInt(dataset.xs.len);
        const accuracy = correct_count / @as(f32, @floatFromInt(dataset.xs.len));
        return .{ .loss = loss, .accuracy = accuracy };
    }

    fn feedforward(self: Self, x: []f32) !struct { zs: [][]f32, as: [][]f32 } {
        const zs: [][]f32 = try self.allocator.alloc([]f32, self.biases.len);
        const as: [][]f32 = try self.allocator.alloc([]f32, self.biases.len);
        for (self.weights, self.biases, zs, as, 0..) |wl, bl, *zl, *al, l| {
            const ak: []f32 = if (l == 0) x else as[l - 1];
            // TODO: Check if every sub-slice in the 2nd dimension of b has the same size
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
            zl.* = try math.add(self.allocator, weighted_sum, bl);
            al.* = try math.sigmoid(self.allocator, zl.*);
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
            utils.free2dMatrix(self.allocator, zs);
            utils.free2dMatrix(self.allocator, as);
        }
        // Backprop
        const d_ws: [][][]f32 = try self.allocator.alloc([][]f32, self.weights.len); // ∂C/∂w
        const d_bs: [][]f32 = try self.allocator.alloc([]f32, self.biases.len); // ∂C/∂b
        var l = d_bs.len - 1;
        while (l >= 0) {
            var dl: []f32 = undefined; // δl
            if (l == d_bs.len - 1) { // Last layer
                const d_al = try costDerivative(self.allocator, as[l], y); // ∇aC
                defer self.allocator.free(d_al);
                const d_zl = try math.sigmoidDerivative(self.allocator, zs[l]); // σ'(zL)
                defer self.allocator.free(d_zl);
                dl = try math.mul(self.allocator, d_al, d_zl); // δL = ∇aC * σ'(zL)
            } else {
                const wl = self.weights[l + 1];
                const dm = d_bs[l + 1];
                const weighted_sum = try self.allocator.alloc(f32, wl.len);
                defer self.allocator.free(weighted_sum);
                for (wl, weighted_sum) |wm, *s| {
                    // NOTE: We don't need to use transpose here since we choose (k, l) as the order of dimensions in the weights
                    // in contrast to (l, k) on the website
                    s.* = try math.dot(self.allocator, wm, dm); // (wm)T * δm
                }
                const d_zl = try math.sigmoidDerivative(self.allocator, zs[l]); // σ'(zl)
                defer self.allocator.free(d_zl);
                dl = try math.mul(self.allocator, weighted_sum, d_zl); // δl = ((wm)T * δm) ⊙ σ'(zl)
            }
            d_bs[l] = dl; // ∂C/∂bl = δl
            const ak = if (l == 0) x else as[l - 1];
            const d_wk = try self.allocator.alloc([]f32, ak.len); // ∂C/∂wl
            for (ak, d_wk) |aki, *d_wl| {
                d_wl.* = try self.allocator.alloc(f32, dl.len);
                for (dl, d_wl.*) |dlj, *d_w| {
                    d_w.* = aki * dlj; // ∂C/∂wl = ak * δl
                }
            }
            d_ws[l] = d_wk;
            if (l > 0) {
                l -= 1;
            } else {
                break;
            }
        }
        return .{ .d_ws = d_ws, .d_bs = d_bs };
    }
};

fn findMaxIndex(a: []f32) usize {
    var max_index: usize = 0;
    var max_value = a[0];
    for (a[1..], 1..) |ai, i| {
        if (ai > max_value) {
            max_index = i;
            max_value = ai;
        }
    }
    return max_index;
}

// Cost function: C = 1/2 * Σ(yi - ai)^2 (mean squared error)
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

// Derivative of cost function: dC/dai = ai - yi
fn costDerivative(allocator: Allocator, a: []f32, y: []f32) ![]f32 {
    if (a.len != y.len) {
        unreachable;
    }
    const dc = try allocator.alloc(f32, a.len);
    for (dc, a, y) |*dci, ai, yi| {
        dci.* = ai - yi;
    }
    return dc;
}
