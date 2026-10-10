//! Independent desired-behavior tests for the PR635 graph-cache repair.
const std = @import("std");
const gc = @import("graph_cache");

const Stream = @typeInfo(@typeInfo(@TypeOf(@as(gc.Body, undefined).run)).pointer.child).@"fn".param_types[1].?;
const stream: Stream = .{ .d = undefined, .handle = null };

const Fake = struct {
    live: usize = 0,
    freed: usize = 0,
    captures: usize = 0,
    launches: usize = 0,
    slot: u8 = 0,

    fn engine(f: *Fake) gc.Engine {
        return .{ .ctx = f, .capture = capture, .launch = launch, .free = free };
    }
    fn capture(ctx: *anyopaque, _: @TypeOf(stream), _: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.captures += 1;
        f.live += 1;
        return &f.slot;
    }
    fn launch(ctx: *anyopaque, _: *anyopaque, _: @TypeOf(stream)) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.launches += 1;
    }
    fn free(ctx: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.freed += 1;
        f.live -= 1;
    }
};

fn body(_: *anyopaque, _: @TypeOf(stream)) anyerror!void {}
var opaque_body: u8 = 0;
const step: gc.Body = .{ .ctx = &opaque_body, .run = body };

const Vote = struct {
    calls: usize = 0,
    fail_on: ?usize = null,
    fail_after_capture: ?*Fake = null,
    fn agree(v: *Vote) gc.Agree {
        return .{ .ctx = v, .all = all };
    }
    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const v: *Vote = @ptrCast(@alignCast(ctx));
        v.calls += 1;
        if (v.fail_after_capture) |f| {
            if (f.captures > 0) return error.VoteFailed;
        }
        if (v.fail_on == v.calls) return error.VoteFailed;
        return ok;
    }
};

test "capture is freed when post-capture Agree throws" {
    var f: Fake = .{};
    // Vote counts differ between versions; inject the error only after the fake engine captures.
    var v: Vote = .{ .fail_after_capture = &f };
    var c = gc.Cache(u32).init(std.testing.allocator, f.engine(), .{});
    c.agree = v.agree();
    try std.testing.expectError(error.VoteFailed, c.run(1, stream, 7, step));
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(usize, 0), c.count());
    c.deinit();
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(usize, 1), f.freed);
}

test "capture is freed when entry allocation fails" {
    var f: Fake = .{};
    var storage: [0]u8 = .{};
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var c = gc.Cache(u32).init(fixed.allocator(), f.engine(), .{});
    try std.testing.expectEqual(gc.Outcome.eager, try c.run(1, stream, 7, step));
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(usize, 0), c.count());
    c.deinit();
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(f.captures, f.freed);
}

const Shared = struct {
    // One slot per rank and collective ordinal; a bounded peer wait makes mismatches fail without hanging.
    offered: [16][2]std.atomic.Value(u8) = std.mem.zeroes([16][2]std.atomic.Value(u8)),
    ready: std.atomic.Value(u32) = .init(0),
    start: std.atomic.Value(bool) = .init(false),
};

const Rank = struct {
    shared: *Shared,
    id: usize,
    calls: usize = 0,

    fn agree(r: *Rank) gc.Agree {
        return .{ .ctx = r, .all = all };
    }
    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const r: *Rank = @ptrCast(@alignCast(ctx));
        const n = r.calls;
        r.calls += 1;
        if (n >= r.shared.offered.len) return error.TooManyCollectives;
        r.shared.offered[n][r.id].store(if (ok) 1 else 2, .release);
        for (0..1_000_000) |_| {
            const peer = r.shared.offered[n][1 - r.id].load(.acquire);
            if (peer != 0) return ok and peer == 1;
            std.Thread.yield() catch {};
        }
        return error.PeerMissing;
    }
};

const Job = struct {
    cache: *gc.Cache(u32),
    shared: *Shared,
    fingerprint: u64,
    outcome: ?gc.Outcome = null,
    failure: ?anyerror = null,

    fn run(j: *Job) void {
        _ = j.shared.ready.fetchAdd(1, .acq_rel);
        while (!j.shared.start.load(.acquire)) std.Thread.yield() catch {};
        j.outcome = j.cache.run(1, stream, j.fingerprint, step) catch |err| {
            j.failure = err;
            return;
        };
    }
};

fn pair(a: *Job, b: *Job) !void {
    a.shared.ready.store(0, .release);
    a.shared.start.store(false, .release);
    var ta = try std.Thread.spawn(.{}, Job.run, .{a});
    var tb = std.Thread.spawn(.{}, Job.run, .{b}) catch |err| {
        // Release the first thread if its peer cannot be created.
        a.shared.start.store(true, .release);
        ta.join();
        return err;
    };
    for (0..1_000_000) |_| {
        if (a.shared.ready.load(.acquire) == 2) break;
        std.Thread.yield() catch {};
    }
    a.shared.start.store(true, .release);
    ta.join();
    tb.join();
}

test "one rank's host allocation failure makes both eager before capture" {
    var shared: Shared = .{};
    var fa: Fake = .{};
    var fb: Fake = .{};
    var ra: Rank = .{ .shared = &shared, .id = 0 };
    var rb: Rank = .{ .shared = &shared, .id = 1 };
    var empty: [0]u8 = .{};
    var no_memory = std.heap.FixedBufferAllocator.init(&empty);
    var a = gc.Cache(u32).init(std.heap.c_allocator, fa.engine(), .{});
    var b = gc.Cache(u32).init(no_memory.allocator(), fb.engine(), .{});
    defer a.deinit();
    defer b.deinit();
    a.agree = ra.agree();
    b.agree = rb.agree();
    var ja: Job = .{ .cache = &a, .shared = &shared, .fingerprint = 7 };
    var jb: Job = .{ .cache = &b, .shared = &shared, .fingerprint = 7 };
    try pair(&ja, &jb);
    try std.testing.expect(ja.failure == null and jb.failure == null);
    try std.testing.expectEqual(gc.Outcome.eager, ja.outcome.?);
    try std.testing.expectEqual(gc.Outcome.eager, jb.outcome.?);
    try std.testing.expectEqual(ra.calls, rb.calls);
    try std.testing.expectEqual(@as(usize, 0), fa.captures);
    try std.testing.expectEqual(@as(usize, 0), fb.captures);
    try std.testing.expectEqual(@as(usize, 0), fa.launches);
    try std.testing.expectEqual(@as(usize, 0), fb.launches);
}

test "fingerprint divergence cannot skip a blocking paired Agree" {
    var shared: Shared = .{};
    var fa: Fake = .{};
    var fb: Fake = .{};
    var ra: Rank = .{ .shared = &shared, .id = 0 };
    var rb: Rank = .{ .shared = &shared, .id = 1 };
    var a = gc.Cache(u32).init(std.heap.c_allocator, fa.engine(), .{});
    var b = gc.Cache(u32).init(std.heap.c_allocator, fb.engine(), .{});
    defer a.deinit();
    defer b.deinit();
    a.agree = ra.agree();
    b.agree = rb.agree();
    var first_a: Job = .{ .cache = &a, .shared = &shared, .fingerprint = 7 };
    var first_b: Job = .{ .cache = &b, .shared = &shared, .fingerprint = 7 };
    try pair(&first_a, &first_b);
    try std.testing.expect(first_a.failure == null and first_b.failure == null);
    try std.testing.expectEqual(gc.Outcome.captured, first_a.outcome.?);
    try std.testing.expectEqual(gc.Outcome.captured, first_b.outcome.?);
    try std.testing.expectEqual(ra.calls, rb.calls);

    // Rank B invalidates; the original rank A replays while B's miss vote cannot find its peer.
    var next_a: Job = .{ .cache = &a, .shared = &shared, .fingerprint = 7 };
    var next_b: Job = .{ .cache = &b, .shared = &shared, .fingerprint = 8 };
    try pair(&next_a, &next_b);
    try std.testing.expect(next_a.failure == null and next_b.failure == null);
    try std.testing.expectEqual(ra.calls, rb.calls);
    try std.testing.expectEqual(gc.Outcome.captured, next_a.outcome.?);
    try std.testing.expectEqual(gc.Outcome.captured, next_b.outcome.?);
    try std.testing.expectEqual(@as(usize, 2), fa.captures);
    try std.testing.expectEqual(@as(usize, 2), fb.captures);
}

test "max zero remains eager through repeated misses" {
    var f: Fake = .{};
    var c = gc.Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 0 });
    defer c.deinit();
    try std.testing.expectEqual(gc.Outcome.eager, try c.run(1, stream, 7, step));
    try std.testing.expectEqual(gc.Outcome.eager, try c.run(2, stream, 7, step));
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(usize, 0), c.count());
}

fn noRoom() bool {
    return false;
}

test "memory-floor recovery retains Settings.max" {
    var f: Fake = .{};
    var c = gc.Cache(u32).init(std.testing.allocator, f.engine(), .{ .max = 2 });
    defer c.deinit();
    c.room = noRoom;
    try std.testing.expectEqual(gc.Outcome.eager, try c.run(1, stream, 7, step));
    try std.testing.expect(c.cap <= c.settings.max);
    c.room = null;
    for (1..4) |key| {
        try std.testing.expectEqual(gc.Outcome.captured, try c.run(@intCast(key), stream, 7, step));
    }
    try std.testing.expect(c.count() <= c.settings.max);
}
