const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn dot(allocator: Allocator, vector1: []f32, vector2: []f32) !f32 {
    const mul_result = try mul(allocator, vector1, vector2);
    defer allocator.free(mul_result);
    var result: f32 = 0;
    for (mul_result) |m| {
        result += m;
    }
    return result;
}

pub fn mul(allocator: Allocator, vector1: []f32, vector2: []f32) ![]f32 {
    if (vector1.len != vector2.len) {
        unreachable;
    }
    const result: []f32 = try allocator.alloc(f32, vector1.len);
    for (result, vector1, vector2) |*r, v1, v2| {
        r.* = v1 * v2;
    }
    return result;
}

pub fn add(allocator: Allocator, vector1: []f32, vector2: []f32) ![]f32 {
    if (vector1.len != vector2.len) {
        unreachable;
    }
    const result: []f32 = try allocator.alloc(f32, vector1.len);
    for (result, vector1, vector2) |*r, v1, v2| {
        r.* = v1 + v2;
    }
    return result;
}

pub fn sigmoid(allocator: Allocator, vector: []f32) ![]f32 {
    const result = try allocator.alloc(f32, vector.len);
    for (result, vector) |*r, v| {
        r.* = 1 / (1 + @exp(-v));
    }
    return result;
}

// Derivative of sigmoid: dσ = σ * (1-σ)
pub fn sigmoidDerivative(allocator: Allocator, vector: []f32) ![]f32 {
    const sigmoid_result = try sigmoid(allocator, vector);
    defer allocator.free(sigmoid_result);
    const result = try allocator.alloc(f32, vector.len);
    for (result, sigmoid_result) |*r, s| {
        r.* = s * (1 - s);
    }
    return result;
}
