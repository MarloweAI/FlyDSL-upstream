// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"

#include "flydsl/Conversion/FlyKtraceToROCDL/FlyKtraceToROCDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyKtrace/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_FLYKTRACETOROCDLCONVERSIONPASS
#include "flydsl/Conversion/FlyKtraceToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly_ktrace;

namespace {

constexpr int64_t kRecordBytes = 32;
constexpr int64_t kOffTimestamp = 0;
constexpr int64_t kOffPayload = 8;
constexpr int64_t kOffHwId = 16;
constexpr int64_t kOffXccId = 20;
constexpr int64_t kOffPacked = 24;
constexpr int64_t kOffStartSlot = 28;

constexpr uint32_t kKindMark = 0;
constexpr uint32_t kKindRangePush = 1;
constexpr uint32_t kKindRangePop = 2;
constexpr uint32_t kKindRangeStart = 3;
constexpr uint32_t kKindRangeStop = 4;

constexpr int64_t kUnpairedSlot = 0xFFFFFFFF;

// Slot 2 sits in the header line next to cursor 0 and is never decoded.
constexpr int64_t kSinkSlot = 2;

constexpr int64_t kDefaultCapacitySlots = (64 << 20) / kRecordBytes - 1;

// Partition cursors are 64-bit, one per 128-byte line, starting at slot 1. Must match
// ktrace_format.CURSOR_STRIDE_BYTES.
constexpr int64_t kCursorStrideBytes = 128;

constexpr int64_t kWaveSize = 64;

struct BufferLayout {
  int64_t capacitySlots;
  int64_t partitions;
  int64_t partitionSlots;
  // Static slots: wave i of a launch owns region i, with no claim atomic; one launch per read.
  bool staticSlots = false;

  int64_t firstRecordSlot() const { return 1 + partitions * (kCursorStrideBytes / kRecordBytes); }
};

struct KernelTraceState {
  Value base;
  Value bufPtr;
  Value hwId;
  Value xccId;
  Value recording;
};

Value i32Const(OpBuilder &b, Location loc, int64_t v) {
  return b.create<LLVM::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(v));
}

Value gepBytes(OpBuilder &b, Location loc, Value ptr, Value byteOffset) {
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext(), 1);
  return b.create<LLVM::GEPOp>(loc, ptrTy, b.getI8Type(), ptr, ValueRange{byteOffset},
                               LLVM::GEPNoWrapFlags::none);
}

Value callIntrinsic(OpBuilder &b, Location loc, StringRef name, Type resultTy,
                    ValueRange args = {}) {
  auto op = b.create<LLVM::CallIntrinsicOp>(loc, resultTy, b.getStringAttr(name), args);
  return op.getResult(0);
}

Value hwregImmarg(OpBuilder &b, Location loc, int64_t regId, int64_t size) {
  return i32Const(b, loc, regId | ((size - 1) << 11));
}

Value i64Const(OpBuilder &b, Location loc, int64_t v) {
  return b.create<LLVM::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(v));
}

// Partition key of this wave: consecutive waves of a row of workgroups get consecutive keys,
// odd multipliers scatter the rows, and the dispatch id rotates successive launches. The key
// needs no memory load; an exact linear index would need the grid size from the kernargs.
Value partitionKey(OpBuilder &b, Location loc, gpu::GPUFuncOp kernel, const BufferLayout &layout,
                   Value dispatchId) {
  Type i32 = b.getI32Type();
  int64_t wavesPerBlock = 1024 / kWaveSize;
  if (std::optional<ArrayRef<int32_t>> known = kernel.getKnownBlockSize())
    wavesPerBlock = llvm::divideCeil((*known)[0] * (*known)[1] * (*known)[2], kWaveSize);

  auto blockId = [&](gpu::Dimension dim) -> Value {
    return b.create<arith::IndexCastOp>(loc, i32, b.create<gpu::BlockIdOp>(loc, dim));
  };
  Value key = b.create<arith::MulIOp>(loc, blockId(gpu::Dimension::z), i32Const(b, loc, 0x85EB));
  key = b.create<arith::AddIOp>(loc, key, blockId(gpu::Dimension::y));
  key = b.create<arith::MulIOp>(loc, key, i32Const(b, loc, 0x9E37));
  Value column =
      b.create<arith::MulIOp>(loc, blockId(gpu::Dimension::x), i32Const(b, loc, wavesPerBlock));
  key = b.create<arith::AddIOp>(loc, key, column);
  Value wave = b.create<arith::IndexCastOp>(loc, i32, b.create<gpu::SubgroupIdOp>(loc, nullptr));
  key = b.create<arith::AddIOp>(loc, key, wave);
  Value dispatchLow = b.create<LLVM::TruncOp>(loc, i32, dispatchId);
  Value rotation = b.create<arith::MulIOp>(loc, dispatchLow,
                                           i32Const(b, loc, static_cast<int32_t>(0x9E3779B1u)));
  key = b.create<arith::AddIOp>(loc, key, rotation);
  return b.create<arith::AndIOp>(loc, key, i32Const(b, loc, layout.partitions - 1));
}

struct LaunchShape {
  Value groupsX, groupsY, groupsZ, wavesPerBlock;
};

// Workgroup counts of the launch from the hidden kernel arguments (code object v5+):
// hidden_block_count_{x,y,z} at +0/+4/+8 and hidden_group_size_{x,y,z} at +12/+14/+16.
LaunchShape launchShape(OpBuilder &b, Location loc, gpu::GPUFuncOp kernel) {
  Type i32 = b.getI32Type();
  auto argsTy = LLVM::LLVMPointerType::get(b.getContext(), 4);
  Value hidden = callIntrinsic(b, loc, "llvm.amdgcn.implicitarg.ptr", argsTy);
  auto hiddenArg = [&](int64_t offset, Type ty) -> Value {
    Value ptr =
        b.create<LLVM::GEPOp>(loc, argsTy, b.getI8Type(), hidden,
                              ValueRange{i32Const(b, loc, offset)}, LLVM::GEPNoWrapFlags::none);
    return b.create<LLVM::LoadOp>(loc, ty, ptr, /*alignment=*/ty.getIntOrFloatBitWidth() / 8,
                                  /*isVolatile=*/false, /*isNonTemporal=*/false,
                                  /*isInvariant=*/true);
  };
  LaunchShape shape{hiddenArg(0, i32), hiddenArg(4, i32), hiddenArg(8, i32), Value()};
  if (std::optional<ArrayRef<int32_t>> known = kernel.getKnownBlockSize()) {
    shape.wavesPerBlock =
        i32Const(b, loc, llvm::divideCeil((*known)[0] * (*known)[1] * (*known)[2], kWaveSize));
  } else {
    Value threads = nullptr;
    for (int64_t offset : {12, 14, 16}) {
      Value dim = b.create<LLVM::ZExtOp>(loc, i32, hiddenArg(offset, b.getI16Type()));
      threads = threads ? b.create<arith::MulIOp>(loc, threads, dim).getResult() : dim;
    }
    shape.wavesPerBlock = b.create<arith::CeilDivUIOp>(loc, threads, i32Const(b, loc, kWaveSize));
  }
  return shape;
}

// Index of this wave among all waves of its launch.
Value linearWaveIndex(OpBuilder &b, Location loc, const LaunchShape &shape) {
  Type i32 = b.getI32Type();
  auto blockId = [&](gpu::Dimension dim) -> Value {
    return b.create<arith::IndexCastOp>(loc, i32, b.create<gpu::BlockIdOp>(loc, dim));
  };
  Value block = b.create<arith::MulIOp>(loc, blockId(gpu::Dimension::z), shape.groupsY);
  block = b.create<arith::AddIOp>(loc, block, blockId(gpu::Dimension::y));
  block = b.create<arith::MulIOp>(loc, block, shape.groupsX);
  block = b.create<arith::AddIOp>(loc, block, blockId(gpu::Dimension::x));
  block = b.create<arith::MulIOp>(loc, block, shape.wavesPerBlock);
  Value wave = b.create<arith::IndexCastOp>(loc, i32, b.create<gpu::SubgroupIdOp>(loc, nullptr));
  return b.create<arith::AddIOp>(loc, block, wave);
}

Value firstActiveLane(OpBuilder &b, Location loc) {
  Type i64 = b.getI64Type();
  Value one = b.create<LLVM::ConstantOp>(loc, b.getI1Type(), b.getIntegerAttr(b.getI1Type(), 1));
  Value mask = b.create<ROCDL::BallotOp>(loc, i64, one);
  Value cttz = b.create<LLVM::CountTrailingZerosOp>(loc, i64, mask, true);
  return b.create<LLVM::TruncOp>(loc, b.getI32Type(), cttz);
}

FailureOr<Value> blockFilter(OpBuilder &b, Location loc, StringRef spec, Value xccId,
                             Operation *diagOp) {
  spec = spec.trim();
  if (spec.empty() || spec == "all")
    return Value();

  if (spec.consume_front("xcc:")) {
    uint32_t want;
    if (spec.getAsInteger(10, want))
      return diagOp->emitError("ktrace: FLYDSL_KTRACE_BLOCKS 'xcc:N' needs an "
                               "integer N, got ")
             << spec;
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, xccId, i32Const(b, loc, want))
        .getResult();
  }

  SmallVector<StringRef, 4> parts;
  spec.split(parts, ',');
  if (parts.size() != 3)
    return diagOp->emitError("ktrace: FLYDSL_KTRACE_BLOCKS must be 'x,y,z', "
                             "'xcc:N' or 'all'; got ")
           << spec;

  static constexpr gpu::Dimension kDims[] = {gpu::Dimension::x, gpu::Dimension::y,
                                             gpu::Dimension::z};
  Type i32 = b.getI32Type();
  Value cond;
  for (auto [dim, part] : llvm::zip_equal(kDims, parts)) {
    uint32_t want;
    if (part.trim().getAsInteger(10, want))
      return diagOp->emitError("ktrace: FLYDSL_KTRACE_BLOCKS 'x,y,z' needs "
                               "integer coordinates, got ")
             << spec;
    Value got = b.create<gpu::BlockIdOp>(loc, dim);
    got = b.create<arith::IndexCastOp>(loc, i32, got);
    Value term =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, got, i32Const(b, loc, want));
    cond = cond ? b.create<arith::AndIOp>(loc, cond, term).getResult() : term;
  }
  return cond;
}

FailureOr<KernelTraceState> emitPrologue(gpu::GPUFuncOp kernel, int64_t eventsPerWave,
                                         const BufferLayout &layout, StringRef blocksSpec) {
  Block &entry = kernel.getBody().front();
  if (entry.getNumArguments() == 0)
    return kernel.emitError("ktrace: kernel has no arguments, so no trace-buffer parameter; the "
                            "kernel was traced without ktrace enabled");

  OpBuilder b(&entry, entry.begin());
  Location loc = kernel.getLoc();
  Type i32Ty = b.getI32Type();

  KernelTraceState st;
  Value rawBuf = entry.getArgument(entry.getNumArguments() - 1);
  auto flyPtrTy = dyn_cast<fly::PointerType>(rawBuf.getType());
  if (!flyPtrTy)
    return kernel.emitError("ktrace: expected the trailing kernel argument to be a "
                            "!fly.ptr trace buffer, got ")
           << rawBuf.getType();

  auto llvmPtrTy = LLVM::LLVMPointerType::get(b.getContext(), 1);
  st.bufPtr = b.create<fly::ToLLVMPtrOp>(loc, llvmPtrTy, rawBuf, b.getI32IntegerAttr(1));

  st.hwId =
      callIntrinsic(b, loc, "llvm.amdgcn.s.getreg", b.getI32Type(), {hwregImmarg(b, loc, 4, 32)});
  st.xccId =
      callIntrinsic(b, loc, "llvm.amdgcn.s.getreg", b.getI32Type(), {hwregImmarg(b, loc, 20, 4)});
  Value dispatchId = callIntrinsic(b, loc, "llvm.amdgcn.dispatch.id", b.getI64Type());
  // Waves that start together claim from different partition cursors: one shared cursor
  // serializes every wave's claim.
  Value key = i32Const(b, loc, 0);
  LaunchShape shape;
  Value waveIndex;
  if (layout.staticSlots) {
    shape = launchShape(b, loc, kernel);
    waveIndex = linearWaveIndex(b, loc, shape);
  } else if (layout.partitions > 1) {
    key = partitionKey(b, loc, kernel, layout, dispatchId);
  }

  Value firstThread;
  for (gpu::Dimension dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    Value block = b.create<gpu::BlockIdOp>(loc, dim);
    Value thread = b.create<gpu::ThreadIdOp>(loc, dim);
    Value blockZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, block,
                                              b.create<arith::ConstantIndexOp>(loc, 0));
    Value threadZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, thread,
                                               b.create<arith::ConstantIndexOp>(loc, 0));
    Value dimZero = b.create<arith::AndIOp>(loc, blockZero, threadZero);
    firstThread =
        firstThread ? b.create<arith::AndIOp>(loc, firstThread, dimZero).getResult() : dimZero;
  }
  auto launchIf = b.create<scf::IfOp>(loc, firstThread, false);
  b.setInsertionPointToStart(launchIf.thenBlock());
  Value launchPtr = gepBytes(b, loc, st.bufPtr, i32Const(b, loc, 4));
  b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, launchPtr, i32Const(b, loc, 1),
                              LLVM::AtomicOrdering::monotonic);
  if (layout.staticSlots) {
    // The host decodes exactly this many wave regions.
    Value waves = b.create<arith::MulIOp>(loc, shape.groupsX, shape.groupsY);
    waves = b.create<arith::MulIOp>(loc, waves, shape.groupsZ);
    waves = b.create<arith::MulIOp>(loc, waves, shape.wavesPerBlock);
    b.create<LLVM::StoreOp>(loc, waves, gepBytes(b, loc, st.bufPtr, i32Const(b, loc, 8)));
  }
  b.setInsertionPointAfter(launchIf);

  auto queuePtrTy = LLVM::LLVMPointerType::get(b.getContext(), 4);
  Value queuePtr = callIntrinsic(b, loc, "llvm.amdgcn.queue.ptr", queuePtrTy);
  Value queueId = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), queuePtr);

  FailureOr<Value> recording = blockFilter(b, loc, blocksSpec, st.xccId, kernel);
  if (failed(recording))
    return failure();
  st.recording = *recording;

  auto emitClaim = [&]() -> Value {
    Value lane = b.create<arith::IndexCastOp>(loc, i32Ty, b.create<gpu::LaneIdOp>(loc, nullptr));
    Value isLeader =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, firstActiveLane(b, loc));
    auto claimIf = b.create<scf::IfOp>(loc, TypeRange{i32Ty}, isLeader, true);
    b.setInsertionPointToStart(claimIf.thenBlock());
    Value inBounds, base;
    if (layout.staticSlots) {
      int64_t regions = (layout.capacitySlots + 1 - layout.firstRecordSlot()) / eventsPerWave;
      inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, waveIndex,
                                         i32Const(b, loc, regions));
      base = b.create<arith::MulIOp>(loc, waveIndex, i32Const(b, loc, eventsPerWave));
      // A wave past the buffer records nothing; the header counts it for the overflow report.
      Value dropped = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, waveIndex,
                                              i32Const(b, loc, regions));
      auto droppedIf = b.create<scf::IfOp>(loc, dropped, false);
      b.setInsertionPointToStart(droppedIf.thenBlock());
      b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, st.bufPtr, i32Const(b, loc, 1),
                                  LLVM::AtomicOrdering::monotonic, "agent");
      b.setInsertionPointAfter(droppedIf);
    } else {
      Value cursorOff = b.create<arith::MulIOp>(loc, key, i32Const(b, loc, kCursorStrideBytes));
      cursorOff = b.create<arith::AddIOp>(loc, cursorOff, i32Const(b, loc, kRecordBytes));
      Value cursor = gepBytes(b, loc, st.bufPtr, cursorOff);
      // The cursor is 64-bit so long captures cannot wrap it back into range. A claim past the
      // partition still advances it, so the host sees the overflow.
      Value claimed = b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, cursor,
                                                  i64Const(b, loc, eventsPerWave),
                                                  LLVM::AtomicOrdering::monotonic, "agent");
      // A partition smaller than one region holds no wave, and the bound below would wrap.
      inBounds =
          layout.partitionSlots < eventsPerWave
              ? b.create<arith::ConstantOp>(loc, b.getBoolAttr(false)).getResult()
              : b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, claimed,
                                        i64Const(b, loc, layout.partitionSlots - eventsPerWave))
                    .getResult();
      // Partitions interleave by wave region: the j-th claim of partition `key` owns region
      // j * partitions + key, so the claimed regions form one prefix the host reads at once.
      Value claimedSlots = b.create<LLVM::TruncOp>(loc, i32Ty, claimed);
      base = b.create<arith::MulIOp>(loc, claimedSlots, i32Const(b, loc, layout.partitions));
      Value keySlots = b.create<arith::MulIOp>(loc, key, i32Const(b, loc, eventsPerWave));
      base = b.create<arith::AddIOp>(loc, base, keySlots);
    }
    base = b.create<arith::AddIOp>(loc, base, i32Const(b, loc, layout.firstRecordSlot()));
    auto metadataIf = b.create<scf::IfOp>(loc, inBounds, false);
    b.setInsertionPointToStart(metadataIf.thenBlock());
    Value byteOff = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, base, i32Const(b, loc, kRecordBytes)),
        i32Const(b, loc, 8));
    b.create<LLVM::StoreOp>(loc, dispatchId, gepBytes(b, loc, st.bufPtr, byteOff));
    Value queueByteOff = b.create<arith::AddIOp>(loc, byteOff, i32Const(b, loc, 8));
    b.create<LLVM::StoreOp>(loc, queueId, gepBytes(b, loc, st.bufPtr, queueByteOff));
    // Remaining header words: (block.z << 8 | wave in block) at +4, block.x at +24, block.y
    // at +28.
    auto blockId = [&](gpu::Dimension dim) -> Value {
      return b.create<arith::IndexCastOp>(loc, i32Ty, b.create<gpu::BlockIdOp>(loc, dim));
    };
    Value wave =
        b.create<arith::IndexCastOp>(loc, i32Ty, b.create<gpu::SubgroupIdOp>(loc, nullptr));
    Value waveWord = b.create<arith::OrIOp>(
        loc, b.create<arith::ShLIOp>(loc, blockId(gpu::Dimension::z), i32Const(b, loc, 8)), wave);
    Value headerOff = b.create<arith::SubIOp>(loc, byteOff, i32Const(b, loc, 4));
    b.create<LLVM::StoreOp>(loc, waveWord, gepBytes(b, loc, st.bufPtr, headerOff));
    Value blockXOff = b.create<arith::AddIOp>(loc, queueByteOff, i32Const(b, loc, 8));
    b.create<LLVM::StoreOp>(loc, blockId(gpu::Dimension::x),
                            gepBytes(b, loc, st.bufPtr, blockXOff));
    Value blockYOff = b.create<arith::AddIOp>(loc, blockXOff, i32Const(b, loc, 4));
    b.create<LLVM::StoreOp>(loc, blockId(gpu::Dimension::y),
                            gepBytes(b, loc, st.bufPtr, blockYOff));
    b.setInsertionPointAfter(metadataIf);
    Value claimedBase =
        b.create<arith::SelectOp>(loc, inBounds, base, i32Const(b, loc, kUnpairedSlot));
    b.create<scf::YieldOp>(loc, ValueRange{claimedBase});
    b.setInsertionPointToStart(claimIf.elseBlock());
    b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, 0)});
    b.setInsertionPointAfter(claimIf);
    return b.create<ROCDL::ReadfirstlaneOp>(loc, i32Ty, claimIf.getResult(0));
  };

  if (!st.recording) {
    st.base = emitClaim();
    return st;
  }

  auto claimIf = b.create<scf::IfOp>(loc, TypeRange{b.getI32Type()}, st.recording, true);
  b.setInsertionPointToStart(claimIf.thenBlock());
  b.create<scf::YieldOp>(loc, ValueRange{emitClaim()});
  b.setInsertionPointToStart(claimIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, 0)});
  st.base = claimIf.getResult(0);
  return st;
}

// The per-wave event counter lives in SSA values instead of the wave's header word, so events
// need no returning atomic. It is threaded through scf.for, scf.while and scf.if. An scf.if
// gives each branch a static index range, so it cannot hold a loop with events. Kernels with
// events anywhere else keep the memory counter.
bool canThreadCounter(gpu::GPUFuncOp kernel, ArrayRef<Operation *> events) {
  if (!kernel.getBody().hasOneBlock() ||
      !isa<gpu::ReturnOp>(kernel.getBody().front().getTerminator()))
    return false;
  for (Operation *op : events) {
    bool inLoop = false;
    for (Operation *p = op->getParentOp(); p != kernel.getOperation(); p = p->getParentOp()) {
      if (isa<scf::ForOp, scf::WhileOp>(p))
        inLoop = true;
      else if (!isa<scf::IfOp>(p) || inLoop)
        return false;
    }
  }
  return true;
}

// Lanes that left a divergent loop early carry a smaller count; the wave consumed the largest.
Value waveMaxCount(OpBuilder &b, Location loc, Value count) {
  return callIntrinsic(b, loc, "llvm.amdgcn.wave.reduce.umax", b.getI32Type(),
                       {count, i32Const(b, loc, 0)});
}

class CounterThreader {
public:
  CounterThreader(ArrayRef<Operation *> events, MLIRContext *ctx)
      : events(events.begin(), events.end()), rewriter(ctx) {}

  // Event op -> its index among the wave's events.
  DenseMap<Operation *, Value> local;

  // Threads `count` through `block` and returns the count at its end.
  Value thread(Block &block, Value count) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (events.contains(&op)) {
        local[&op] = count;
        OpBuilder b(op.getContext());
        b.setInsertionPointAfter(&op);
        count = b.create<arith::AddIOp>(op.getLoc(), count, i32Const(b, op.getLoc(), 1));
      } else if (auto loop = dyn_cast<scf::ForOp>(op); loop && hasEvent(loop)) {
        count = threadFor(loop, count);
      } else if (auto loop = dyn_cast<scf::WhileOp>(op); loop && hasEvent(loop)) {
        count = threadWhile(loop, count);
      } else if (auto branch = dyn_cast<scf::IfOp>(op); branch && hasEvent(branch)) {
        count = threadIf(branch, count);
      }
    }
    return count;
  }

private:
  bool hasEvent(Operation *root) { return countEvents(root->getRegions()) > 0; }

  int64_t countEvents(MutableArrayRef<Region> regions) {
    int64_t n = 0;
    for (Region &region : regions)
      region.walk([&](Operation *op) { n += events.contains(op); });
    return n;
  }

  // A branch takes indices only when some lane of the wave runs it, so a branch no lane runs
  // costs no slot. The lanes of a divergent wave run both branches, which get disjoint ranges.
  Value threadIf(scf::IfOp branch, Value count) {
    Location loc = branch.getLoc();
    OpBuilder b(branch);
    auto used = [&](Value cond, int64_t events) -> Value {
      if (events == 0)
        return i32Const(b, loc, 0);
      Value lanes = b.create<ROCDL::BallotOp>(loc, b.getI64Type(), cond);
      Value runs =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lanes, i64Const(b, loc, 0));
      return b.create<arith::SelectOp>(loc, runs, i32Const(b, loc, events), i32Const(b, loc, 0));
    };
    Value cond = branch.getCondition();
    Value elseStart =
        b.create<arith::AddIOp>(loc, count, used(cond, countEvents(branch.getThenRegion())));
    thread(*branch.thenBlock(), count);
    Block *elseBlock = branch.elseBlock();
    if (!elseBlock)
      return elseStart;
    thread(*elseBlock, elseStart);
    Value notCond =
        b.create<arith::XOrIOp>(loc, cond, b.create<arith::ConstantOp>(loc, b.getBoolAttr(true)));
    return b.create<arith::AddIOp>(loc, elseStart,
                                   used(notCond, countEvents(branch.getElseRegion())));
  }

  Value threadFor(scf::ForOp loop, Value count) {
    rewriter.setInsertionPoint(loop);
    auto threaded = cast<scf::ForOp>(*loop.replaceWithAdditionalYields(
        rewriter, ValueRange{count}, /*replaceInitOperandUsesInLoop=*/false,
        [](OpBuilder &, Location, ArrayRef<BlockArgument> args) {
          return SmallVector<Value>{args.front()};
        }));
    Value end = thread(*threaded.getBody(), threaded.getRegionIterArgs().back());
    Operation *yield = threaded.getBody()->getTerminator();
    yield->setOperand(yield->getNumOperands() - 1, end);
    OpBuilder b(threaded->getContext());
    b.setInsertionPointAfter(threaded);
    return waveMaxCount(b, threaded.getLoc(), threaded.getResults().back());
  }

  Value threadWhile(scf::WhileOp loop, Value count) {
    Location loc = loop.getLoc();
    Type i32 = count.getType();
    SmallVector<Value> inits(loop.getInits());
    inits.push_back(count);
    SmallVector<Type> types(loop.getResultTypes());
    types.push_back(i32);
    rewriter.setInsertionPoint(loop);
    auto threaded = rewriter.create<scf::WhileOp>(loc, types, inits);
    threaded.getBefore().takeBody(loop.getBefore());
    threaded.getAfter().takeBody(loop.getAfter());

    Block &before = threaded.getBefore().front();
    Value beforeEnd = thread(before, before.addArgument(i32, loc));
    threaded.getConditionOp().getArgsMutable().append(beforeEnd);
    Block &after = threaded.getAfter().front();
    Value afterEnd = thread(after, after.addArgument(i32, loc));
    cast<scf::YieldOp>(after.getTerminator()).getResultsMutable().append(afterEnd);

    rewriter.replaceOp(loop, threaded.getResults().drop_back());
    OpBuilder b(threaded->getContext());
    b.setInsertionPointAfter(threaded);
    return waveMaxCount(b, loc, threaded.getResults().back());
  }

  DenseSet<Operation *> events;
  IRRewriter rewriter;
};

// Stores the wave's attempted event count in its header word, which the host checks for
// events dropped past the per-wave limit.
void storeAttemptedCount(gpu::ReturnOp ret, const KernelTraceState &st, Value count,
                         int64_t capacitySlots) {
  OpBuilder b(ret);
  Location loc = ret.getLoc();
  Type i32 = b.getI32Type();
  Value lane = b.create<arith::IndexCastOp>(loc, i32, b.create<gpu::LaneIdOp>(loc, nullptr));
  Value write =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, firstActiveLane(b, loc));
  if (st.recording)
    write = b.create<arith::AndIOp>(loc, write, st.recording);
  Value claimed = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, st.base,
                                          i32Const(b, loc, capacitySlots));
  write = b.create<arith::AndIOp>(loc, write, claimed);
  auto writeIf = b.create<scf::IfOp>(loc, write, false);
  b.setInsertionPointToStart(writeIf.thenBlock());
  Value headerOff = b.create<arith::MulIOp>(loc, st.base, i32Const(b, loc, kRecordBytes));
  b.create<LLVM::StoreOp>(loc, count, gepBytes(b, loc, st.bufPtr, headerOff));
}

// Scalar record stores stay in the scalar cache until the wave writes it back.
void writeBackScalarCache(gpu::ReturnOp ret) {
  OpBuilder b(ret);
  b.create<LLVM::InlineAsmOp>(ret.getLoc(), TypeRange{}, ValueRange{},
                              "s_dcache_wb\n\ts_waitcnt lgkmcnt(0)", "", /*has_side_effects=*/true,
                              /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
                              LLVM::AsmDialectAttr(), ArrayAttr());
}

struct EventShape {
  uint32_t kind;
  uint32_t eventId;
  Value payload;
  Value startSlot;
  bool producesSlot;
  uint32_t payloadType = 0;
};

// Lowers an event whose index comes from the register counter. Every value is wave-uniform,
// so all active lanes store the same record and the event adds no divergent branch; a record
// that does not fit goes to a sink slot in the header line instead of being skipped.
Value expandUniformEvent(Operation *op, const EventShape &shape, const KernelTraceState &st,
                         int64_t eventsPerWave, int64_t capacitySlots, Value local) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Type i32 = b.getI32Type();

  scf::IfOp recordingIf;
  if (st.recording) {
    SmallVector<Type, 1> results;
    if (shape.producesSlot)
      results.push_back(i32);
    recordingIf = b.create<scf::IfOp>(loc, results, st.recording, shape.producesSlot);
    b.setInsertionPointToStart(recordingIf.thenBlock());
  }

  Value ts = callIntrinsic(b, loc, "llvm.amdgcn.s.memrealtime", b.getI64Type());
  Value slot = b.create<arith::AddIOp>(loc, st.base, local);
  Value capacity = i32Const(b, loc, capacitySlots);
  Value ok = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, st.base, capacity);
  Value withinWave = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local,
                                             i32Const(b, loc, eventsPerWave - 1));
  ok = b.create<arith::AndIOp>(loc, ok, withinWave);
  Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, slot, capacity);
  ok = b.create<arith::AndIOp>(loc, ok, inBounds);
  Value slot1 = b.create<arith::AddIOp>(loc, slot, i32Const(b, loc, 1));
  Value token;
  if (shape.producesSlot)
    token = b.create<arith::SelectOp>(loc, ok, slot1, i32Const(b, loc, kUnpairedSlot));
  Value recordSlot = b.create<arith::SelectOp>(loc, ok, slot1, i32Const(b, loc, kSinkSlot));
  Value byteOff = b.create<arith::MulIOp>(loc, recordSlot, i32Const(b, loc, kRecordBytes));
  Value recPtr = gepBytes(b, loc, st.bufPtr, byteOff);

  // The documented payload is the first active lane's value.
  Value payload =
      shape.payload
          ? b.create<ROCDL::ReadfirstlaneOp>(loc, shape.payload.getType(), shape.payload)
                .getResult()
          : b.create<LLVM::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(0)).getResult();
  Value startSlot = shape.startSlot ? shape.startSlot : i32Const(b, loc, 0);
  uint32_t packed = (shape.eventId & 0xFFFFFF) | ((shape.kind & 0x7) << 24) |
                    (shape.payloadType << 27) | (shape.payload ? (1u << 31) : 0);
  // The record is wave-uniform, so scalar stores keep it out of VGPRs. A block holding inline
  // asm keeps its exec-zero skip, so a branch that no lane runs stores nothing. A scalar store
  // can read its SGPRs after it issues, and the compiler may reuse them right after the asm, so
  // the asm waits for its stores.
  auto pair = VectorType::get({2}, i32);
  auto quad = VectorType::get({4}, i32);
  Value timeWords = b.create<LLVM::BitcastOp>(loc, pair, ts);
  Value payloadWords = b.create<LLVM::BitcastOp>(loc, pair, payload);
  Value timePayload =
      b.create<LLVM::ShuffleVectorOp>(loc, timeWords, payloadWords, ArrayRef<int32_t>{0, 1, 2, 3});
  Value ids = b.create<LLVM::PoisonOp>(loc, quad);
  Value idWords[] = {st.hwId, st.xccId, i32Const(b, loc, static_cast<int64_t>(packed)), startSlot};
  for (auto [i, word] : llvm::enumerate(idWords))
    ids = b.create<LLVM::InsertElementOp>(loc, ids, word, i32Const(b, loc, i));
  static_assert(kOffTimestamp == 0 && kOffPayload == 8 && kOffHwId == 16 && kOffXccId == 20 &&
                    kOffPacked == 24 && kOffStartSlot == 28,
                "the scalar record stores write the record layout as two 16-byte halves");
  b.create<LLVM::InlineAsmOp>(
      loc, TypeRange{}, ValueRange{timePayload, ids, recPtr},
      "s_store_dwordx4 $0, $2, 0x0\n\ts_store_dwordx4 $1, $2, 0x10\n\ts_waitcnt lgkmcnt(0)",
      "s,s,s", /*has_side_effects=*/true, /*is_align_stack=*/false,
      LLVM::tailcallkind::TailCallKind::None, LLVM::AsmDialectAttr(), ArrayAttr());

  Value result = token;
  if (recordingIf && shape.producesSlot) {
    b.create<scf::YieldOp>(loc, ValueRange{token});
    b.setInsertionPointToStart(recordingIf.elseBlock());
    b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, kUnpairedSlot)});
    result = recordingIf.getResult(0);
  }
  if (result)
    op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return result;
}

Value expandEvent(Operation *op, const EventShape &shape, const KernelTraceState &st,
                  int64_t eventsPerWave, int64_t capacitySlots) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Type i32 = b.getI32Type();

  Value lane = b.create<arith::IndexCastOp>(loc, i32, b.create<gpu::LaneIdOp>(loc, nullptr));
  Value leader = firstActiveLane(b, loc);
  Value isLeader = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, leader);
  if (st.recording)
    isLeader = b.create<arith::AndIOp>(loc, isLeader, st.recording);

  SmallVector<Type, 1> guardResults;
  if (shape.producesSlot)
    guardResults.push_back(i32);
  auto leaderIf = b.create<scf::IfOp>(loc, guardResults, isLeader, shape.producesSlot);
  b.setInsertionPointToStart(leaderIf.thenBlock());

  Value baseInBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, st.base,
                                               i32Const(b, loc, capacitySlots));
  auto baseIf = b.create<scf::IfOp>(loc, guardResults, baseInBounds, shape.producesSlot);
  b.setInsertionPointToStart(baseIf.thenBlock());

  // Read the clock before the slot atomic so its latency does not skew the timestamp.
  Value ts = callIntrinsic(b, loc, "llvm.amdgcn.s.memrealtime", b.getI64Type());
  Value ctrPtr = gepBytes(b, loc, st.bufPtr,
                          b.create<arith::MulIOp>(loc, st.base, i32Const(b, loc, kRecordBytes)));
  Value local = b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, ctrPtr,
                                            i32Const(b, loc, 1), LLVM::AtomicOrdering::monotonic);
  Value slot = b.create<arith::AddIOp>(loc, st.base, local);

  Value withinWave = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local,
                                             i32Const(b, loc, eventsPerWave - 1));
  Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, slot,
                                           i32Const(b, loc, capacitySlots));
  Value storeOk = b.create<arith::AndIOp>(loc, withinWave, inBounds);

  auto storeIf = b.create<scf::IfOp>(loc, guardResults, storeOk, shape.producesSlot);
  b.setInsertionPointToStart(storeIf.thenBlock());

  Value slot1 = b.create<arith::AddIOp>(loc, slot, i32Const(b, loc, 1));
  Value byteOff = b.create<arith::MulIOp>(loc, slot1, i32Const(b, loc, kRecordBytes));
  Value recPtr = gepBytes(b, loc, st.bufPtr, byteOff);

  Value payload =
      shape.payload
          ? shape.payload
          : b.create<LLVM::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(0)).getResult();
  Value startSlot = i32Const(b, loc, 0);
  if (shape.startSlot) {
    assert(shape.startSlot.getType() == i32 &&
           "range_stop's token must already be lowered to its i32 slot");
    startSlot = shape.startSlot;
  }
  uint32_t packed = (shape.eventId & 0xFFFFFF) | ((shape.kind & 0x7) << 24) |
                    (shape.payloadType << 27) | (shape.payload ? (1u << 31) : 0);

  struct Field {
    int64_t offset;
    Value value;
  };
  Field fields[] = {
      {kOffTimestamp, ts},
      {kOffPayload, payload},
      {kOffHwId, st.hwId},
      {kOffXccId, st.xccId},
      {kOffPacked, i32Const(b, loc, static_cast<int64_t>(packed))},
      {kOffStartSlot, startSlot},
  };
  for (const Field &f : fields) {
    Value fieldPtr = gepBytes(b, loc, recPtr, i32Const(b, loc, f.offset));
    b.create<LLVM::StoreOp>(loc, f.value, fieldPtr);
  }

  if (!shape.producesSlot) {
    op->erase();
    return Value();
  }

  b.create<scf::YieldOp>(loc, ValueRange{slot1});

  b.setInsertionPointToStart(storeIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, kUnpairedSlot)});

  b.setInsertionPointAfter(storeIf);
  b.create<scf::YieldOp>(loc, ValueRange{storeIf.getResult(0)});

  b.setInsertionPointToStart(baseIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, kUnpairedSlot)});

  b.setInsertionPointAfter(baseIf);
  b.create<scf::YieldOp>(loc, ValueRange{baseIf.getResult(0)});

  b.setInsertionPointToStart(leaderIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, kUnpairedSlot)});

  Value result = leaderIf.getResult(0);
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return result;
}

LogicalResult checkTargetArch(gpu::GPUModuleOp mod, StringRef chipHint) {
  auto rejectChip = [&](StringRef chip) {
    return mod.emitError("ktrace: in-kernel timestamps are not supported on "
                         "target arch '")
           << chip
           << "'; supported: gfx942 (CDNA3) and gfx950 (CDNA4). s_memrealtime "
              "is absent on RDNA, and s_getreg silently reads a different "
              "register there rather than failing";
  };

  ArrayAttr targets = mod.getTargetsAttr();
  if (!targets || targets.empty()) {
    if (chipHint.empty())
      return mod.emitError("ktrace: gpu.module carries no target and no "
                           "fly_ktrace.chip; cannot verify that in-kernel "
                           "timestamps are supported");
    if (chipHint.starts_with("gfx942") || chipHint.starts_with("gfx950"))
      return success();
    return rejectChip(chipHint);
  }

  bool sawRocdl = false;
  for (Attribute t : targets) {
    auto rocdlTarget = dyn_cast<ROCDL::ROCDLTargetAttr>(t);
    if (!rocdlTarget)
      continue;
    StringRef chip = rocdlTarget.getChip();
    if (!chip.starts_with("gfx942") && !chip.starts_with("gfx950"))
      return rejectChip(chip);
    sawRocdl = true;
  }
  if (!sawRocdl)
    return mod.emitError("ktrace: gpu.module has no ROCDL target");
  return success();
}

struct SelectTokenConversion : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getType());
    if (!converted)
      return failure();
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, converted, adaptor.getCondition(),
                                                 adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

LogicalResult legalizeTokenTypes(Operation *root) {
  MLIRContext *ctx = root->getContext();
  Type i32 = IntegerType::get(ctx, 32);

  TypeConverter converter;
  converter.addConversion([](Type t) { return t; });
  converter.addConversion([i32](fly_ktrace::TokenType) { return i32; });
  auto materialize = [](OpBuilder &b, Type resultType, ValueRange inputs, Location loc) -> Value {
    return b.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
  };
  converter.addSourceMaterialization(materialize);
  converter.addTargetMaterialization(materialize);

  ConversionTarget target(*ctx);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  scf::populateSCFStructuralTypeConversionTarget(converter, target);

  target.addDynamicallyLegalOp<arith::SelectOp>(
      [&converter](arith::SelectOp op) { return converter.isLegal(op.getType()); });

  RewritePatternSet patterns(ctx);
  scf::populateSCFStructuralTypeConversions(converter, patterns);
  patterns.add<SelectTokenConversion>(converter, ctx);
  if (failed(applyPartialConversion(root, target, std::move(patterns))))
    return failure();

  SmallVector<UnrealizedConversionCastOp> dead;
  root->walk([&](UnrealizedConversionCastOp cast) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
      return;
    if (!cast.getOperand(0).getType().isInteger(32))
      return;
    if (!isa<fly_ktrace::TokenType>(cast.getResult(0).getType()))
      return;
    cast.getResult(0).replaceAllUsesWith(cast.getOperand(0));
    dead.push_back(cast);
  });
  for (UnrealizedConversionCastOp cast : dead)
    cast.erase();
  return success();
}

struct FlyKtraceToROCDLConversionPass
    : public impl::FlyKtraceToROCDLConversionPassBase<FlyKtraceToROCDLConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();

    if (failed(legalizeTokenTypes(root))) {
      signalPassFailure();
      return;
    }

    llvm::StringMap<uint32_t> ids;
    uint32_t nextId = 1;
    if (auto known = root->getAttrOfType<DictionaryAttr>("fly_ktrace.known_event_names")) {
      for (NamedAttribute kv : known) {
        auto v = dyn_cast<IntegerAttr>(kv.getValue());
        if (!v)
          continue;
        uint32_t id = static_cast<uint32_t>(v.getInt());
        ids[kv.getName().strref()] = id;
        nextId = std::max(nextId, id + 1);
      }
    }

    auto idFor = [&ids, &nextId](StringRef name) {
      auto it = ids.find(name);
      if (it != ids.end())
        return it->second;
      uint32_t id = nextId++;
      ids[name] = id;
      return id;
    };

    WalkResult result = root->walk([&](gpu::GPUFuncOp kernel) {
      SmallVector<Operation *> events;
      kernel.walk([&](Operation *op) {
        if (isa<MarkOp, RangePushOp, RangePopOp, RangeStartOp, RangeStopOp, InactiveTokenOp>(op))
          events.push_back(op);
      });
      if (events.empty())
        return WalkResult::advance();

      auto mod = kernel->getParentOfType<gpu::GPUModuleOp>();
      StringRef chipHint;
      if (auto attr = root->getAttrOfType<StringAttr>("fly_ktrace.chip"))
        chipHint = attr.getValue();
      if (!mod || failed(checkTargetArch(mod, chipHint)))
        return WalkResult::interrupt();

      constexpr int64_t kEventsPerWave = 256;

      BufferLayout layout{kDefaultCapacitySlots, /*partitions=*/1, /*partitionSlots=*/0};
      if (auto attr = root->getAttrOfType<IntegerAttr>("fly_ktrace.capacity_slots"))
        layout.capacitySlots = attr.getInt();
      if (auto attr = root->getAttrOfType<IntegerAttr>("fly_ktrace.partitions"))
        layout.partitions = attr.getInt();
      if (auto attr = root->getAttrOfType<IntegerAttr>("fly_ktrace.partition_slots"))
        layout.partitionSlots = attr.getInt();
      else
        layout.partitionSlots = (layout.capacitySlots + 1 - layout.firstRecordSlot()) /
                                layout.partitions / kEventsPerWave * kEventsPerWave;
      if (auto attr = root->getAttrOfType<StringAttr>("fly_ktrace.slots"))
        layout.staticSlots = attr.getValue() == "static";
      if (layout.partitions < 1 || !llvm::isPowerOf2_64(layout.partitions) ||
          layout.partitionSlots < 0 || layout.partitionSlots % kEventsPerWave != 0 ||
          layout.firstRecordSlot() + layout.partitions * layout.partitionSlots >
              layout.capacitySlots + 1) {
        kernel.emitError("ktrace: fly_ktrace.partitions / fly_ktrace.partition_slots do not fit "
                         "the trace buffer of ")
            << layout.capacitySlots << " slots";
        return WalkResult::interrupt();
      }

      StringRef blocksSpec;
      if (auto attr = root->getAttrOfType<StringAttr>("fly_ktrace.blocks"))
        blocksSpec = attr.getValue();

      FailureOr<KernelTraceState> st = emitPrologue(kernel, kEventsPerWave, layout, blocksSpec);
      if (failed(st))
        return WalkResult::interrupt();

      SmallVector<Operation *> slotEvents;
      for (Operation *op : events)
        if (!isa<InactiveTokenOp>(op))
          slotEvents.push_back(op);
      DenseMap<Operation *, Value> counter;
      if (canThreadCounter(kernel, slotEvents)) {
        Block &entry = kernel.getBody().front();
        OpBuilder b(&entry, entry.begin());
        CounterThreader threader(slotEvents, kernel.getContext());
        Value count = threader.thread(entry, i32Const(b, kernel.getLoc(), 0));
        auto ret = cast<gpu::ReturnOp>(entry.getTerminator());
        storeAttemptedCount(ret, *st, count, layout.capacitySlots);
        writeBackScalarCache(ret);
        counter = std::move(threader.local);
      }

      for (Operation *op : events) {
        if (isa<InactiveTokenOp>(op)) {
          OpBuilder b(op);
          Value seed = i32Const(b, op->getLoc(), kUnpairedSlot);
          op->getResult(0).replaceAllUsesWith(seed);
          op->erase();
          continue;
        }

        EventShape shape;
        if (auto mark = dyn_cast<MarkOp>(op)) {
          shape = {kKindMark, idFor(mark.getEventName()), mark.getPayload(), Value(), false};
        } else if (auto push = dyn_cast<RangePushOp>(op)) {
          shape = {kKindRangePush, idFor(push.getEventName()), push.getPayload(), Value(), false};
        } else if (auto pop = dyn_cast<RangePopOp>(op)) {
          shape = {kKindRangePop, 0, pop.getPayload(), Value(), false};
        } else if (auto start = dyn_cast<RangeStartOp>(op)) {
          shape = {kKindRangeStart, idFor(start.getEventName()), start.getPayload(), Value(), true};
        } else {
          auto end = cast<RangeStopOp>(op);
          shape = {kKindRangeStop, idFor(end.getEventName()), end.getPayload(), end.getToken(),
                   false};
        }
        if (shape.eventId > 0xFFFFFF) {
          op->emitError("ktrace: more than 2^24 - 1 distinct event names");
          return WalkResult::interrupt();
        }
        if (auto type = op->getAttrOfType<StringAttr>("payload_type")) {
          int encoding = llvm::StringSwitch<int>(type.getValue())
                             .Case("uint64", 0)
                             .Case("int64", 1)
                             .Case("float32", 2)
                             .Case("float64", 3)
                             .Case("bool", 4)
                             .Default(-1);
          if (encoding < 0) {
            op->emitError("ktrace: unsupported payload_type ") << type.getValue();
            return WalkResult::interrupt();
          }
          shape.payloadType = encoding;
        }
        if (Value local = counter.lookup(op))
          expandUniformEvent(op, shape, *st, kEventsPerWave, layout.capacitySlots, local);
        else
          expandEvent(op, shape, *st, kEventsPerWave, layout.capacitySlots);
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (!ids.empty()) {
      OpBuilder b(root->getContext());
      SmallVector<NamedAttribute> entries;
      entries.reserve(ids.size());
      for (const auto &kv : ids)
        entries.emplace_back(b.getStringAttr(kv.getKey()), b.getI32IntegerAttr(kv.getValue()));
      root->setAttr("fly_ktrace.event_names", b.getDictionaryAttr(entries));
    }
  }
};

} // namespace
