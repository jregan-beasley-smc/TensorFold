//! Model-free CUDA qualification of the unchanged PR635 arena, memory hook and graph cache.

const std = @import("std");
const cuda = @import("cuda");
const MiB = 1 << 20;

fn require(ok: bool) !void {
    if (!ok) return error.QualificationFailed;
}

fn checkBytes(b: cuda.DeviceBuffer, value: u8) !void {
    var out: [4096]u8 = undefined;
    try b.download(0, &out);
    for (out) |x| try require(x == value);
}

const CopyBody = struct {
    src: cuda.DeviceBuffer,
    dst: cuda.DeviceBuffer,
    inject_failure: bool = false,
    calls: usize = 0,

    fn run(ctx: *anyopaque, s: cuda.Stream) anyerror!void {
        const b: *CopyBody = @ptrCast(@alignCast(ctx));
        b.calls += 1;
        try b.src.fill8(0x5a, s.handle);
        if (b.inject_failure) {
            b.inject_failure = false;
            return error.InjectedCaptureBodyFailure;
        }
        try b.dst.copyFrom(0, b.src.ptr, b.src.len, s.handle);
    }

    fn body(b: *CopyBody) cuda.graph_cache.Body {
        return .{ .ctx = b, .run = run };
    }
};

fn graphChecks(d: *const cuda.Driver, s: cuda.Stream) !void {
    var src = try cuda.DeviceBuffer.alloc(d, 4096);
    defer src.free();
    var dst = try cuda.DeviceBuffer.alloc(d, 4096);
    defer dst.free();
    var other = try cuda.DeviceBuffer.alloc(d, 4096);
    defer other.free();
    var body: CopyBody = .{ .src = src, .dst = dst };
    var engine: cuda.graph_cache.CudaEngine = .{ .d = d };
    var cache = cuda.graph_cache.Cache(u32).init(std.heap.c_allocator, engine.engine(), .{ .max = 2, .min_keep = 1 });
    defer cache.deinit();
    const fp = cuda.graph_cache.fingerprintOf(&.{ src.ptr, dst.ptr, src.len });
    try require(try cache.run(1, s, fp, body.body()) == .captured);
    try s.synchronize();
    try checkBytes(dst, 0x5a);
    try dst.fill8(0, s.handle);
    try require(try cache.run(1, s, fp, body.body()) == .replayed);
    try s.synchronize();
    try checkBytes(dst, 0x5a);
    try require(body.calls == 1);
    try require(try cache.run(2, s, fp, body.body()) == .captured);
    try require(try cache.run(1, s, fp, body.body()) == .replayed);
    try require(try cache.run(3, s, fp, body.body()) == .captured);
    try s.synchronize();
    try require(cache.count() == 2 and cache.has(1) and cache.has(3) and !cache.has(2));
    try require(cache.stats.evicted == 1);
    body.dst = other;
    const changed = cuda.graph_cache.fingerprintOf(&.{ src.ptr, other.ptr, src.len });
    try require(changed != fp);
    try require(try cache.run(1, s, changed, body.body()) == .captured);
    try s.synchronize();
    try checkBytes(other, 0x5a);
    try require(cache.count() == 1 and cache.stats.dropped == 2);
    cache.dropAll();
    try require(cache.count() == 0);

    var bad = cuda.graph_cache.Cache(u32).init(std.heap.c_allocator, engine.engine(), .{ .max = 2 });
    defer bad.deinit();
    body.inject_failure = true;
    try require(try bad.run(4, s, changed, body.body()) == .eager);
    try s.synchronize();
    try require(bad.failed and bad.count() == 0);
    try require(try cuda.graph.captureStatus(s) == .none);
    try checkBytes(other, 0x5a);
    try other.fill8(0, s.handle);
    try require(try bad.run(5, s, changed, body.body()) == .eager);
    try s.synchronize();
    try checkBytes(other, 0x5a);
}

fn arenaChecks(d: *const cuda.Driver, s: cuda.Stream, backend: cuda.arena.Backend) !void {
    var a = cuda.arena.Arena.init(std.heap.c_allocator, backend, .{ .small_max = 8192, .small_chunk = 2 * MiB });
    defer a.deinit();
    try require(cuda.memory.currentHook() == null);
    var before = try cuda.DeviceBuffer.alloc(d, 4096);
    var before_live = true;
    defer if (before_live) before.free();
    a.install();
    defer cuda.memory.setHook(null);
    try require(a.alloc(0) == null);
    var x = try cuda.DeviceBuffer.alloc(d, 4096);
    var x_live = true;
    defer if (x_live) x.free();
    var y = try cuda.DeviceBuffer.alloc(d, 4096);
    defer y.free();
    try require(x.ptr % 512 == 0 and y.ptr % 512 == 0 and x.ptr + x.len <= y.ptr);
    try require(!a.free(before.ptr));
    before.free();
    before_live = false;
    try x.fill8(0xa1, s.handle);
    try y.fill8(0xb2, s.handle);
    try s.synchronize();
    try checkBytes(x, 0xa1);
    try checkBytes(y, 0xb2);
    const counted = cuda.memory.usage(false).device;
    var borrowed = x;
    borrowed.borrowed = true;
    borrowed.free();
    try require(cuda.memory.usage(false).device == counted);
    try checkBytes(x, 0xa1);
    try require(try x.at(x.len) == x.ptr + x.len);
    if (x.at(x.len + 1)) |_| return error.UncheckedBounds else |e| try require(e == error.Invalid);
    const one: [1]u8 = .{1};
    if (x.upload(x.len, &one)) |_| return error.UncheckedBounds else |e| try require(e == error.Invalid);
    if (x.copyFrom(4095, y.ptr, 2, s.handle)) |_| return error.UncheckedBounds else |e| try require(e == error.Invalid);
    const old_ptr = x.ptr;
    x.free();
    x_live = false;
    var reused = try cuda.DeviceBuffer.alloc(d, 4096);
    defer reused.free();
    try require(reused.ptr == old_ptr and a.stats.reused == 1);
    try checkBytes(y, 0xb2);
    var large = try cuda.DeviceBuffer.alloc(d, 16384);
    large.free();
    try require(a.large.count() == 0 and a.stats.large == 1);
    try graphChecks(d, s);
    try require(a.stats.chunk_bytes <= 16 * MiB);
}

pub fn main(init: std.process.Init) !void {
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var s = try cuda.Stream.init(&d, true);
    defer s.deinit();
    const before = cuda.memory.usage(false).device;
    var plain: cuda.arena.Plain = .{ .d = &d };
    try arenaChecks(&d, s, plain.backend());
    try require(cuda.memory.currentHook() == null and cuda.memory.usage(false).device == before);
    var vmm_status: []const u8 = "unsupported";
    if (cuda.arena.Vmm.open(&d, ctx.device, 16 * MiB)) |opened| {
        var v = opened;
        defer v.close();
        if (v.granularity <= 8 * MiB and v.va_len <= 16 * MiB) {
            try arenaChecks(&d, s, v.backend());
            vmm_status = "passed";
        } else vmm_status = "skipped_granularity";
    } else |err| {
        if (err != error.Unsupported) return err;
    }
    try require(cuda.memory.currentHook() == null and cuda.memory.usage(false).device == before);
    const Decline = struct {
        allocs: usize = 0,
        frees: usize = 0,
        fn alloc(p: *anyopaque, _: usize) ?u64 {
            const t: *@This() = @ptrCast(@alignCast(p));
            t.allocs += 1;
            return null;
        }
        fn free(p: *anyopaque, _: u64) bool {
            const t: *@This() = @ptrCast(@alignCast(p));
            t.frees += 1;
            return false;
        }
    };
    var declined: Decline = .{};
    cuda.memory.setHook(.{ .ctx = &declined, .alloc = Decline.alloc, .free = Decline.free });
    defer cuda.memory.setHook(null);
    var normal = try cuda.DeviceBuffer.alloc(&d, 4096);
    var normal_live = true;
    errdefer if (normal_live) normal.free();
    try normal.fill8(0xc3, s.handle);
    try s.synchronize();
    try checkBytes(normal, 0xc3);
    normal.free();
    normal_live = false;
    try require(cuda.memory.usage(false).device == before and declined.allocs == 1 and declined.frees == 1);
    cuda.memory.setHook(null);
    var buf: [512]u8 = undefined;
    const result = try std.fmt.bufPrint(&buf, "{{\"schema\":\"pr635_model_free_cuda_v1\",\"plain\":\"passed\",\"graphs\":\"passed\",\"vmm\":\"{s}\",\"requested_payload_bound_bytes\":{d},\"context_reset\":false}}\n", .{ vmm_status, 16 * MiB });
    try std.Io.File.stdout().writeStreamingAll(init.io, result);
}
