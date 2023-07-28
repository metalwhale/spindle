const std = @import("std");
const utils = @import("utils.zig");
const Dataset = @import("dataset.zig").Dataset;
const Allocator = std.mem.Allocator;

const TRAIN_X_FILE_NAME = "train-images-idx3-ubyte";
const TRAIN_Y_FILE_NAME = "train-labels-idx1-ubyte";
const VAL_X_FILE_NAME = "t10k-images-idx3-ubyte";
const VAL_Y_FILE_NAME = "t10k-labels-idx1-ubyte";

pub const Mnist = struct {
    const Self = @This();
    train_xs: [][]f32,
    train_ys: [][]f32,
    val_xs: [][]f32,
    val_ys: [][]f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator) !Mnist {
        const mnist = Mnist{
            .train_xs = undefined,
            .train_ys = undefined,
            .val_xs = undefined,
            .val_ys = undefined,
            .allocator = allocator,
        };
        return mnist;
    }

    pub fn deinit(self: Self) void {
        defer {
            utils.free2dMatrix(self.allocator, self.train_xs);
            utils.free2dMatrix(self.allocator, self.train_ys);
            utils.free2dMatrix(self.allocator, self.val_xs);
            utils.free2dMatrix(self.allocator, self.val_ys);
        }
    }

    pub fn readData(self: *Self, dir_path: []const u8) !struct { train_dataset: Dataset, val_dataset: Dataset } {
        self.train_xs = try readIdxFile(
            self.allocator,
            try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ dir_path, TRAIN_X_FILE_NAME }),
        );
        self.train_ys = try readIdxFile(
            self.allocator,
            try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ dir_path, TRAIN_Y_FILE_NAME }),
        );
        self.val_xs = try readIdxFile(
            self.allocator,
            try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ dir_path, VAL_X_FILE_NAME }),
        );
        self.val_ys = try readIdxFile(
            self.allocator,
            try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ dir_path, VAL_Y_FILE_NAME }),
        );
        const train_dataset = Dataset.init(self.allocator, self.train_xs, self.train_ys);
        const val_dataset = Dataset.init(self.allocator, self.val_xs, self.val_ys);
        return .{ .train_dataset = train_dataset, .val_dataset = val_dataset };
    }
};

// See: http://yann.lecun.com/exdb/mnist/
fn readIdxFile(allocator: Allocator, file_path: []const u8) ![][]f32 {
    var file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();
    const reader = file.reader();
    const dimensions = (try reader.readBytesNoEof(4))[3];
    const samples_len = convertToU32(try reader.readBytesNoEof(4));
    var size: usize = 1;
    var is_x = false;
    if (dimensions > 1) {
        is_x = true;
        for (1..dimensions) |_| {
            size *= convertToU32(try reader.readBytesNoEof(4));
        }
    } else {
        size = 10;
    }
    const data = try allocator.alloc([]f32, samples_len);
    for (data) |*sample| {
        sample.* = try allocator.alloc(f32, size);
        if (is_x) {
            const buffer = try allocator.alloc(u8, size);
            defer allocator.free(buffer);
            _ = try reader.read(buffer);
            for (sample.*, buffer) |*s, b| {
                s.* = @floatFromInt(b);
                if (is_x) {
                    s.* /= 255;
                }
            }
        } else {
            const label = try reader.readByte();
            for (sample.*, 0..) |*s, i| {
                s.* = if (i == label) 1 else 0;
            }
        }
    }
    return data;
}

// TODO: Use builtin functions (?)
fn convertToU32(bytes: [4]u8) u32 {
    var value: u32 = 0;
    for (0..bytes.len) |i| {
        value += @as(u32, bytes[bytes.len - i - 1]) << @as(u5, @truncate(i * 8));
    }
    return value;
}
