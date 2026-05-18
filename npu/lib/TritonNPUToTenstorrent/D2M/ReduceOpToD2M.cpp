#include "PatternTritonNPUToD2M.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "npu/include/Dialect/TritonTenstorrent/IR/Attributes.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
using namespace tt;
namespace triton {
namespace npu {
namespace experimental {

#define DEBUG_TYPE "convert-triton-npu-to-d2m"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// Returns true when the ReduceOp's combiner region contains exactly one
// arith.addf (i.e. the reduction is a simple sum).
static bool isSingleAddfBody(triton::ReduceOp op) {
  Block &body = op.getCombineOp().front();
  unsigned nonReturnOps = 0;
  for (Operation &innerOp : body) {
    if (isa<triton::ReduceReturnOp>(&innerOp))
      continue;
    if (!isa<arith::AddFOp>(&innerOp))
      return false;
    ++nonReturnOps;
  }
  return nonReturnOps == 1;
}

// See
// https://triton-lang.org/main/python-api/triton.language.html#reduction-ops.
struct ConvertReduceOp : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Only handle single-source reduces with an arith.addf combiner body.
    if (adaptor.getOperands().size() != 1)
      return rewriter.notifyMatchFailure(op, "expected single-operand reduce");
    if (!isSingleAddfBody(op))
      return rewriter.notifyMatchFailure(op,
                                         "only arith.addf body is supported");

    int32_t axis = static_cast<int32_t>(op.getAxis());
    Value srcMemRef = adaptor.getOperands()[0];
    auto srcMemRefType = cast<MemRefType>(srcMemRef.getType());
    int64_t rank = srcMemRefType.getRank();
    auto tileType = cast<ttcore::TileType>(srcMemRefType.getElementType());

    LDBG("ReduceOp: src=" << srcMemRefType << " axis=" << axis);

    // Map the Triton reduction axis to a D2M ReduceDim value.
    // Matches TTIRToD2M::dimArgAsReduceDim convention:
    //   last dim  (rank-1) -> ReduceDim::R
    //   second-to-last (rank-2) -> ReduceDim::C
    d2m::ReduceDim reduceDim;
    if (axis == rank - 1)
      reduceDim = d2m::ReduceDim::R;
    else if (axis == rank - 2)
      reduceDim = d2m::ReduceDim::C;
    else
      return rewriter.notifyMatchFailure(
          op, "only last two dims supported for tile-level reduce");

    auto reduceDimAttr = d2m::ReduceDimAttr::get(ctx, reduceDim);

    // Output shape: source shape with the reduced axis removed.
    SmallVector<int64_t> outShape;
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        outShape.push_back(srcMemRefType.getShape()[i]);

    // TODO: zero-initialise the output; for now we rely on hardware zeroing.
    auto outCBLayout = ttcore::CBLayoutAttr::get(outShape, tileType,
                                                 /*buffers=*/outShape.size());
    auto outMemRefType = MemRefType::get(outShape, tileType, outCBLayout,
                                         srcMemRefType.getMemorySpace());
    Value outMemRef =
        memref::AllocOp::create(rewriter, loc, outMemRefType).getResult();

    // Indexing maps: src uses identity; out drops the reduced dim.
    AffineMap srcMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<AffineExpr> outExprs;
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        outExprs.push_back(getAffineDimExpr(i, ctx));
    AffineMap outMap = AffineMap::get(rank, /*symbolCount=*/0, outExprs, ctx);

    SmallVector<AffineMap> indexingMaps = {srcMap, outMap};
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);
    iterTypes[axis] = utils::IteratorType::reduction;

    // Emit linalg.generic iterating over the tile grid;
    // d2m.tile_reduce_sum semantics: result = sum<dim>(A * B, C)
    // with B = tile_fill(1.0) as a unity scaler.
    linalg::GenericOp::create(
        rewriter, loc,
        /*resultTypes=*/TypeRange{},
        /*inputs=*/ValueRange{srcMemRef},
        /*outputs=*/ValueRange{outMemRef}, indexingMaps, iterTypes,
        [&](OpBuilder &b_, Location innerLoc, ValueRange args) {
          // args[0] = src tile (A); args[1] = out tile (C accumulator)
          Value one =
              arith::ConstantOp::create(b_, innerLoc, b_.getF32FloatAttr(1.0f))
                  .getResult();
          Value scalerTile =
              d2m::TileFillOp::create(b_, innerLoc, tileType, one).getResult();
          Value result = d2m::TileReduceSumOp::create(
                             b_, innerLoc, args[1].getType(), args[0],
                             scalerTile, args[1], reduceDimAttr)
                             .getResult();
          linalg::YieldOp::create(b_, innerLoc, result);
        });

    rewriter.replaceOp(op, outMemRef);
    return success();
  }
};

} // namespace

void populateReduceOpConversionPattern(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<ConvertReduceOp>(typeConverter, patterns.getContext());
}

} // namespace experimental
} // namespace npu
} // namespace triton
} // namespace mlir
